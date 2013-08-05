import re
import math


def getwords(doc):
    splitter = re.compile('\\W*')
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]
    return dict([(w, 1) for w in words])


def sample_train(cl):
    cl.add_training_example('Nobody owns the water.', 'good')
    cl.add_training_example('the quick rabbit jumps fences', 'good')
    cl.add_training_example('buy pharmaceuticals now', 'bad')
    cl.add_training_example('make quick money at the online casino', 'bad')
    cl.add_training_example('the quick brown fox jumps', 'good')


class Classifier(object):
    def __init__(self, get_features, filename=None):
        """Construct classifier from example file and feature extractor."""
        # Counts of (feature, category) combinations
        self.fc = {}
        # Counts of docs in each category
        self.cc = {}
        # Method for extracting features given an example.
        self.get_features = get_features
        self.thresholds = {}

    def incr_feature(self, feature, category):
        """Incr count of feature, category pair."""
        self.fc.setdefault(feature, {})
        self.fc[feature].setdefault(category, 0)
        self.fc[feature][category] += 1

    def incr_category(self, category):
        """Incr count of number examples in a category."""
        self.cc.setdefault(category, 0)
        self.cc[category] += 1

    def get_feature_count(self, feature, category):
        """Get the total number of examples """
        if feature in self.fc and category in self.fc[feature]:
            return float(self.fc[feature][category])
        return 0.0

    def get_category_count(self, category):
        """Get number of examples in a category."""
        return float(self.cc.get(category, 0.0))

    def get_total_count(self):
        """Get number of examples."""
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def add_training_example(self, example, category):
        features = self.get_features(example)

        # Increase (feature, category) counts to add weight
        # the extracted features.
        for f in features:
            self.incr_feature(f, category)

        # Increase count of examples in this category.
        self.incr_category(category)

    def prob_feat_given_cat(self, feature, category):
        """Find P(feature | category) i.e. probability feature occurs in category."""
        if self.get_category_count(category) == 0:
            return 0

        # The total number of times the feature appeared in this category
        # divided by the total number of examples in the category.
        return self.get_feature_count(feature, category) / self.get_category_count(category)

    def weighted_prob(self, feature, category, p_feat_given_cat_fn, weight=1.0, prior_prob=0.5):
        basic_prob = p_feat_given_cat_fn(feature, category)

        # Find total count of given feature in all categories in order
        # to re-weight using the specified weight and prior.
        total = sum([self.get_feature_count(feature, c) for c in self.categories()])
        reweighted_prob = ((weight * prior_prob) + (total * basic_prob)) / (weight + total)
        return reweighted_prob

    def classify(self, example, default=None):
        probs = {}

        # Find category with max prob
        max_prob = 0.0
        best_cat = None
        for category in self.categories():
            probs[category] = self.prob_cat_given_ex(example, category)

            if probs[category] > max_prob:
                max_prob = probs[category]
                best_cat = category

        for cat, prob in probs.iteritems():
            if cat == best_cat:
                continue
            if probs[cat] * self.get_threshold(best_cat) > probs[best_cat]:
                return default
        return best_cat

    def set_threshold(self, category, thresh):
        self.thresholds[category] = thresh

    def get_threshold(self, category):
        return self.thresholds.get(category, 1.0)


class NaiveBayes(Classifier):

    def prob_ex_given_cat(self, example, category):
        features = self.get_features(example)

        p = 1
        for feature in features:
            p *= self.weighted_prob(feature, category, self.prob_feat_given_cat)
        return p

    def prob_cat_given_ex(self, example, category):
        cat_prob = self.get_category_count(category) / self.get_total_count()
        ex_prob = self.prob_ex_given_cat(example, category)
        return cat_prob * ex_prob


class FisherClassifier(Classifier):

    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.minimums = {}

    def set_minimum(self, category, min_val):
        self.minimums[cat] = min_val

    def get_minimum(self, category):
        return self.minimums.get(category, 0.0)

    def category_prob(self, feature, category):
        """Return the prob the specified feature belongs in the specified
        category, assuminmg an equal number of examples per category.
        """
        clf = self.prob_feat_given_cat(feature, category)
        if clf == 0:
            return 0

        cat_sum = sum([self.prob_feat_given_cat(feature, c) for c in self.categories()])
        p = clf / cat_sum
        return p

    def fisher_prob(self, example, category):
        """If the probabilities are independent and random, then the fisher
        probability will fit a chi-squared distribution.  You would expect
        an item that doesn't belong in a category to contain words of varying
        feature probabilities for that category (random)."""
        p = 1
        features = self.get_features(example)
        for f in features:
            p *= (self.weighted_prob(f, category, self.category_prob))

        fscore = -2 * math.log(p)
        return self.invchi2(fscore, len(features) * 2)

    def invchi2(self, chi, df):
        m = chi / 2.0
        total = term = math.exp(-m)
        for i in range(1, df / 2):
            term *= m / i
            total += term
        return min(total, 1.0)

    def classify(self, example, default=None):
        best = default
        max_val = 0.0
        for c in self.categories():
            p = self.fisher_prob(example, c)
            if p > self.get_minimum(c) and p > max_val:
                best = c
                max_val = p
        return best


