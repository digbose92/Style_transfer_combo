# Style_transfer_combo

Pytorch based implementations of style transfer algorithms . 

## Completed:
- Implementation of the basic algorithm mentioned in Leon Gatys's paper https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

- GPU support enabled .

## Running the code:
- The style transfer code can be run using `run_style_transfer.py` as follows:
```python
python run_style_transfer.py &mdash; style_path '../images/wheat_field_van_gogh.jpg' &mdash; content_path '../images/santa_monica.jpg' &mdash; result_dir '../images' ---max_epochs 60
```
 

## Results:

*Style Image*:

![wheat-field-with-cypresses- 1889 -vincent-van-gogh-met](https://user-images.githubusercontent.com/14272549/43407061-456f9600-943b-11e8-997d-d146d696db70.jpg)

 
*Content Image*:

 ![santa_monica_beach_new](https://user-images.githubusercontent.com/14272549/43407234-c7a1ab22-943b-11e8-833d-dad7a5825e60.jpg)

 After running the optimization on a NVIDIA 940 MX GPU for 60 epochs (ran out of GPU memory beyond that) the style transfer result after transferring the van-gogh style to the content image:
 
 ![post_process_100](https://user-images.githubusercontent.com/14272549/43407294-f4d8b02c-943b-11e8-8d64-aeb28c38ac08.png)


## TODO:

- Enabling optimization run on CPU.. Device selection capability.
- Training the model with coco images
- Automatic weight selection
- Photorealistic image generation 
