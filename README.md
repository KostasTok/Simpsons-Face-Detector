# Simpsons-Face-Detector

A simple face detector for the Simpsons Characters. Used on seasons 27 and 28 it created a database of 4885 images. 

Each image:
<ul>
  <li>is focused on a single face</li>
  <li>has size 200x200</li>
  <li>is paired with a simplified version of itself, which only presents the skin and eyes of the detected face.</li>
</ul>
<div align='center' min-width=820>
  <img src='img.png' width=400 float='left'>
  <img src='faces.png' width=400 float='right' >
</div>


<code>get_faces.ipynb</code> gets the link of a directory where the video files are stored and extracts 150 images from each episode. Subsequently, it calls <code>find_face()</code> from <code>face_detector.py</code> which applies a variety of filters on each image in order to detect a face.

The process is based mainly on identifying clusters of white pixels that look like eyes, because the faces of the characters do not follow a consistent enough pattern.
