# The final goal is to take the query image and find the book
# cover in the database with the best match

from __future__ import print_function
from bookcovers.coverdescriptor import CoverDiscriptor
from bookcovers.covermatcher import CoverMatcher
import argparse
import glob # to grab the path to the book cover images
import csv
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True, help = "path to the book database")
ap.add_argument("-c", "--covers", required = True, help = "path to the directory that contains our book covers")
ap.add_argument("-q", "--query", required = True, help = "path to the query book cover")
ap.add_argument("-s", "--sift", type = int, default = 0, help = "whether or not sift should be used")
args = vars(ap.parse_args())

db = {} # This dictionary has been defined to hadle the information
# of building the database of the book
# it is updated with filename of book as the key and title and author
# of the book as value

for l in csv.reader(open(args["db"])):
    db[l[0]] = l[1: ]

useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.7
minMatches = 40

if useSIFT:
    minMatches = 50

cd = CoverDiscriptor(useSIFT = useSIFT) ########useSIFT
cv = CoverMatcher(cd, glob.glob(args["covers"] + "/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)

queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)

results = cv.search(queryKps, queryDescs)

cv2.imshow("Query", queryImage)

if len(results) == 0:
    print("I could not find a match for that cover!")
    cv2.waitKey(0)

else:
    for (i, (score, coverPath)) in enumerate(results):
        (author, title) = db[coverPath[coverPath.rfind("/") + 1:]]
        print("{}. {:.2f}% : {} - {}".format(i + 1, score * 100, author, title))

        result = cv2.imread(coverPath)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
