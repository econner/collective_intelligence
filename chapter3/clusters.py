from math import sqrt
import random

from PIL import Image
from PIL import ImageDraw


def readfile(filename):
    lines = [line for line in file(filename)]

    # First line is column header
    col_names = line[0].strip().split('\t')[1:]
    row_names = []
    data = []

    for line in lines[1:]:
        p = line.strip().split('\t')
        # First column in each row is row name
        row_names.append(p[0])
        # Data for row is remainder of row
        data.append([float(x) for x in p[1:]])

    return row_names, col_names, data


def pearson(v1, v2):
    sum1 = sum(v1)
    sum2 = sum(v2)

    sum_squares1 = sum([pow(v, 2) for v in v1])
    sum_squares2 = sum([pow(v, 2) for v in v2])

    inner_prod = sum([v1[i] * v2[i] for i in range(len(v1))])

    num = inner_prod - (sum1 * sum2 / len(v1))
    den = sqrt(
        (sum_squares1 - pow(sum1, 2) / len(v1)) *
        (sum_squares2 - pow(sum2, 2) / len(v1))
    )

    if den == 0:
        return 0

    return 1.0 - float(num) / den


class BiCluster(object):

    def __init__(self, vec, left=None, right=None,
                 distance=0.0, class_id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.class_id = class_id
        self.distance = distance


def hcluster(rows, distance=pearson):
    distances = {}
    current_cluster_id = -1

    clust = [BiCluster(rows[i], class_id=i)
             for i in range(len(rows))]

    while len(clust) > 1:
        lowest_pair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                clust_id = (clust[i].class_id, clust[j].class_id)
                if clust_id not in distances:
                    distances[clust_id] = distance(clust[i].vec,
                                                   clust[j].vec)

                d = distances[clust_id]
                if d < closest:
                    closest = d
                    lowest_pair = (i, j)

        # avg the two clusters
        c1 = clust[lowest_pair[0]]
        c2 = clust[lowest_pair[1]]
        merged_vec = map(lambda x: (x[0] + x[1]) / 2,
                         zip(c1.vec, c2.vec))

        new_cluster = BiCluster(merged_vec, left=c1, right=c2,
                                distance=closest,
                                class_id=current_cluster_id)
        current_cluster_id -= 1

        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]

        clust.append(new_cluster)

    return clust[0]


def get_height(clust):
    if clust.left is None and clust.right is None:
        return 1

    return get_height(clust.left) + get_height(clust.right)


def get_depth(clust):
    if clust.left is None and clust.right is None:
        return 0

    return (max(get_depth(clust.left), get_depth(clust.right))
            + clust.distance)


def draw_dendogram(clust, labels, jpeg='clusters.jpg'):
    height = get_height(clust) * 20
    width = 1200
    depth = get_depth(clust)

    scaling = float(width - 150) / depth
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, height / 2, 10, height / 2), fill=(255, 0, 0))

    draw_node(draw, clust, 10, height / 2, scaling, labels)
    img.save(jpeg, 'PNG')


def draw_node(draw, clust, x, y, scaling, labels):
    if clust.class_id < 0:
        h1 = get_height(clust.left) * 20
        h2 = get_height(clust.right) * 20

        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2

        l1 = clust.distance * scaling

        # Vertical line from cluster to children
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2),
                  fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x, bottom - h2 / 2, x + l1, bottom - h2 / 2),
                  fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x, top + h1 / 2, x + l1, top + h1 / 2),
                  fill=(255, 0, 0))

        draw_node(draw, clust.left, x + l1, top + h1 / 2,
                  scaling, labels)
        draw_node(draw, clust.right, x + l1, bottom - h2 / 2,
                  scaling, labels)
    else:
        draw.text((x + 5, y - 7), labels[clust.class_id],
                  (0, 0, 0))


def kcluster(rows, distance=pearson, k=4):
    # Determine min and max vals for each point
    ranges = [(min([row[i] for row in rows]),
               max([row[i] for row in rows]))
              for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0])
                 + ranges[i][0] for i in range(len(rows[0]))]
                for j in range(k)]

    last_matches = None
    for t in range(100):
        print "Iteration %d" % t
        best_matches = [[] for i in range(k)]

        # Find which centroid is closted for each row
        for j, row in enumerate(rows):
            best_match = 0

            for i in range(k):
                d = distance(clusters[i], row)

                if d < distance(clusters[best_match], row):
                    best_match = i

            best_matches[best_match].append(j)

        if best_matches == last_matches:
            break
        last_matches = best_matches

        # Move centroids to avg of members
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(best_matches[i]) > 0:
                for row_id in best_matches[i]:
                    for m, val in enumerate(rows[row_id]):
                        avgs[m] += val
                for j in range(len(avgs)):
                    avgs[j] /= len(best_matches[i])
                clusters[i] = avgs

    return best_matches


def scaledown(data, distance=pearson, rate=0.01):
    n = len(data)

    real_dist = [[distance(data[i], data[j]) for j in range(n)]
                 for i in range(0, n)]

    loc = [[random.random(), random.random()] for i in range(n)]
    fake_dist = [[0.0 for j in range(n)] for i in range(n)]

    last_error = None
    for m in range(0, 1000):
        for i in range(n):
            for j in range(n):
                fake_dist[i][j] = sqrt(
                    sum([pow(loc[i][x] - loc[j][x], 2)
                         for x in range(len(loc[i]))])
                )

        # Move points
        grad = [[0.0, 0.0] for i in range(n)]

        total_error = 0
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                # The error is percent difference between the distances
                errorterm = (fake_dist[j][k] - real_dist[j][k]) / real_dist[j][k]

                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0] += ((loc[k][0] - loc[j][0]) / fake_dist[j][k]) * errorterm
                grad[k][1] += ((loc[k][1] - loc[j][1]) / fake_dist[j][k]) * errorterm

                # Keep track of the total error
                total_error += abs(errorterm)

        print total_error

        # If the answer got worse by moving the points, we are done
        if last_error and last_error < total_error:
            break
        last_error = total_error

        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0] -= rate * grad[k][0]
            loc[k][1] -= rate * grad[k][1]

    return loc


def draw_2d(data, labels, jpeg='mds2d.jpg'):
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) * 1000
        y = (data[i][1] + 0.5) * 1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'PNG')
