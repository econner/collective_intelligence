import re
import math


def getwords(doc):
    splitter = re.compile('\\W*')
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]
    return dict([(w, 1) for w in words])


class Classifier(object):
    def __init__(self, get_features, filename=None):
        # Counts of feature / category combinations
        self.fc = {}
        # Counts of docs in each category
        self.cc = {}
        self.get_features = get_features

    def incr_feature(self, feature, category):
        """Incr count of feature, category pair."""
        self.fc.setdefault(feature, {})
        self.fc[feature].setdefault(category, 0)
        self.fc[feature][category] += 1

    def incr_category(self, category):
        """Incr count of a category."""
        self.cc.setdefault(category, 0)
        self.cc[category] += 1

    def get_feature_count(self, feature, category):
        """Get count of feature in category."""
        if feature in self.fc and category in self.fc[feature]:
            return float(self.fc[feature][category])
        return 0.0

    def get_category_count(self, category):
        """Get number of items in a category."""
        return float(self.cc.get(category, 0.0))

    def get_total_count(self):
        """Get number of examples."""
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def train(self, example, category):
        features = self.get_features(example)

        # Inc count for every feature with this category
        for f in features:
            self.incr_feature(f, category)

        # Inc count for this category
        self.incr_category(category)
