# JamaicanDancer_GenderClassification
Do gender classification on dancer dataset, which consist of motion capture data of the Jamaican dancers.

There are two forms of features, Angle values of 17 joint and symmetry features calculated
from angle.

Angle values:
17(joint) x 3,000(frames) x 3(d.v.a) x 172(subjects)
where ground truth is perceived gender score(range from 0 to 1)

Symmetry features:
21 H (x,y,z) and 30 V, 30 T, 21 R,21 G(with 50 different time window size)
where ground truth is gender label(1 or 2)

