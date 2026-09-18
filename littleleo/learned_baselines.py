"""Bag-of-words baselines, mandatory and non-negotiable.

These exist because a corpus once scored 1.000 for a 22M transformer and 1.000
for TF-IDF plus logistic regression, and only the second number revealed that
the first meant nothing. Word counts are the control group. If they match the
encoder, the encoder is dead weight and the honest move is to ship the
vectoriser.

They are deliberately given every advantage — word and bigram features,
sublinear term frequency, a generous regularisation setting, fitted on the full
training split. A control that has been quietly hobbled proves nothing, and the
temptation to hobble it grows in exact proportion to how much one wants the
transformer to win.

Two of them, because they fail differently: logistic regression optimises a
probabilistic objective and tends to hedge on unfamiliar input, while a linear
SVM maximises margin and tends to commit. On the audit split, where clause
wordings are new, that difference is informative.
"""

from __future__ import annotations

from typing import Sequence

from .schema import Example, Risk, Route, SITE_ACT, SITE_PRE


class TfidfBaseline:
    """One vectoriser per call site, one linear classifier per head.

    Site-separated on purpose: routing is decided by the user's turn and risk by
    the proposed action, and a single model over both would be judged on a
    mixture of two tasks it was never asked to do at once.
    """

    def __init__(self, kind: str = "logreg", name: str = ""):
        self.kind = kind
        self.name = name or f"tfidf_{kind}"
        self._route_vec = None
        self._route_clf = None
        self._risk_vec = None
        self._risk_clf = None

    def _make_classifier(self):
        if self.kind == "linearsvc":
            from sklearn.svm import LinearSVC

            return LinearSVC(C=1.0)
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, C=5.0)

    def _make_vectoriser(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    def fit(self, rows: Sequence[Example]) -> "TfidfBaseline":
        pre = [r for r in rows if r.site == SITE_PRE]
        act = [r for r in rows if r.site == SITE_ACT]

        if pre:
            self._route_vec = self._make_vectoriser()
            features = self._route_vec.fit_transform([r.text for r in pre])
            self._route_clf = self._make_classifier()
            self._route_clf.fit(features, [int(r.route) for r in pre])

        if act:
            self._risk_vec = self._make_vectoriser()
            features = self._risk_vec.fit_transform([r.action for r in act])
            self._risk_clf = self._make_classifier()
            self._risk_clf.fit(features, [int(r.risk) for r in act])
        return self

    def __call__(self, example: Example) -> tuple[Route, Risk]:
        route = Route.LARGE
        if self._route_clf is not None and example.site == SITE_PRE:
            vector = self._route_vec.transform([example.text])
            route = Route(int(self._route_clf.predict(vector)[0]))

        risk = Risk.P2_READONLY
        if self._risk_clf is not None and example.site == SITE_ACT:
            vector = self._risk_vec.transform([example.action])
            risk = Risk(int(self._risk_clf.predict(vector)[0]))
        return route, risk

    def predict_many(self, rows: Sequence[Example]) -> list[tuple[Route, Risk]]:
        """Batched, because per-row ``transform`` calls dominate the runtime."""
        results: list[tuple[Route, Risk]] = [(Route.LARGE, Risk.P2_READONLY)] * len(rows)
        pre = [(i, r) for i, r in enumerate(rows) if r.site == SITE_PRE]
        act = [(i, r) for i, r in enumerate(rows) if r.site == SITE_ACT]
        if pre and self._route_clf is not None:
            predicted = self._route_clf.predict(
                self._route_vec.transform([r.text for _, r in pre]))
            for (index, _), value in zip(pre, predicted):
                results[index] = (Route(int(value)), Risk.P2_READONLY)
        if act and self._risk_clf is not None:
            predicted = self._risk_clf.predict(
                self._risk_vec.transform([r.action for _, r in act]))
            for (index, _), value in zip(act, predicted):
                results[index] = (Route.SMALL, Risk(int(value)))
        return results


def fitted_baselines(train_rows: Sequence[Example]) -> dict[str, TfidfBaseline]:
    """The control group, fitted. Always run; never quietly dropped."""
    return {
        "tfidf_logreg": TfidfBaseline("logreg").fit(train_rows),
        "tfidf_linearsvc": TfidfBaseline("linearsvc").fit(train_rows),
    }


__all__ = ["TfidfBaseline", "fitted_baselines"]
