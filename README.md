# Image Colourizer
Kate Allsebrook 

101142664

Welcome to my project! I hope your marking is going well!
This project is a rudimentary image colourizer, details can
be found in my [report](https://docs.google.com/document/d/1ZT98bGkPrcdLA2NphA-XV8AL4pPSYfLxdYKAAlqMWOk/edit?usp=sharing)

### Getting Started
1. Clone this repo
2. Run `pip install -r requirements.txt`
3. Unzip `model.h5.zip` in the same directory*
4. Run `playground.py`*

*These steps may take a bit of time since `model.h5` is a large file

### Testing Given Images
I've provided 3 test images in the folder `images/`. 
On download, the `playground.py` file shows one test image. To test more images, simply
uncomment the `colourize` function with the name of the file you'd like to test.
The options are:
* `beach.jpg`
* `lighter.jpg`
* `woman_with_hat.jpg`

### Testing Your Own Image
1. Make sure that your image is greyscale, and 64x64
2. Place it in the `images/greyscsale` folder
3. Run `colourize(model, <filename>, show_ground_truth=False)`

Enjoy!



