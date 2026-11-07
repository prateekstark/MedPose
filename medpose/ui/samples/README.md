# Synthetic pose fixture

`synthetic-adult.png` was generated with the image-generation tool on 2026-11-06
for this project. It depicts an invented adult in a frontal standing pose. It is
not a photograph of a real person and contains no supplied patient image.

The setup command makes a two-second MP4 by repeating this image at 10 fps.
The clip tests video decoding, temporal inference and scrubbing; it is not a
walking sequence and does not establish motion-estimation accuracy. No exact
anatomical angles are known for this image. Browser tests assert detection and
finite coordinates, never an accuracy score. Hand-computed geometric checks use
separate artificial landmark coordinates.
