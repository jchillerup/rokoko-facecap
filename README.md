# rokoko-facecap
This project is a fork of trishume's awesome [eyeLike](https://github.com/trishume/eyeLike) for tracking eyes and pupils. For the Rokoko project it's been extended to track facial markers as well, for tracking facial expressions of actors.

The actor is rigged with face paint in 21 places, and the coordinates of those will be streamed to our Unity backend over OSC using [liblo](http://liblo.sourceforge.net/).

![test](https://raw.githubusercontent.com/jchillerup/rokoko-facecap/master/res/jc.jpg)

## Requirements
* [CvBlob](https://code.google.com/p/cvblob)
* [liblo](http://liblo.sourceforge.net/)
