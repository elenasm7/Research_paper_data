# Solar Energy/Irrandiance Prediction Using Multi-Input, Mixed Data Tensorflow Models

This project was initiated by my interet in green energy and ML applications in the Space. After doing more research into where ML/data science work is missing in the fight against climate change. The research brought me to [Climate Change AI](climatechange.ai), by Andrew Ng and others. After reading over the white paper and digging deeper into noted research, I decided to pursue the space of prediciting solar irradiance using sky images at the site of a PV array/panel. 

A big shout out to the groups and people who have made this research even possible. Thank you Hugo T. C. Pedro, David P. Larson, and Carlos F. M. Coimbra for their work [to collect data](https://zenodo.org/record/2826939#.X_x4GelKjlx) for others to do research in this space (J. Renewable Sustainable Energy 11, 036102 (2019); https://doi.org/10.1063/1.5094494).

Below I will detail the work on this project thus far and next steps that I am continuously working on.

## Approach

**Data source**

listed above, but this data was gathered from:
- [Journal of Renewable and Sustainable Energy](https://zenodo.org/record/2826939#.X_x4GelKjlx)
- [Climate Change AI](climatechane.ai)

**Mathodology**
1. Conduct Research
    * Look into areas where research is laking accoring to the work done by the team at Climate Change AI
    * Go through cited research in the relevant space
    * Find data sources to use for models

2. Data Preparation and Preprocessing
    * Create filepaths for images based on Tar.BZ2 files
    * match image file to correct instance in DataFrame
    * get correct time intervals and cross join for earlier irradiance data
    * Test-train split (80-20 split)
    * Scale Data between [0,1] and [-1,1]

3. Image processing and loading

4. Build Multi-input, Mixed Data Model
    * Use pretrain CNN, without the top
    * Create top layers of CNN branch
    * Define simple MLP
    * Create final multi-input layer to merge both branches

5. Train Model

**Next steps**
1. Choose best pretrained model for CNN branch
2. Optimize model by testing different parameter
3. If need be, add in more data
4. Train with all data on cloud platfrom

## Models

Lets start with the Pretrain CNN models we are utilizing for our image branch. The four models we are testing with are:
1. RestNet50
2. InceptionV3
3. Inception RestNet V2
4. VGG 16

The below graphs offer some insight into model accuracy vs. operations needed for one pass. This helped us decide in choosing Inception and Resnet Models

<img src="https://github.com/elenasm7/Research_paper_data/blob/main/model_accuracy_to_G_ops.jpg" atl="accuracy to num of operations" width="600" height="400" />


Below we will get into the basics of why we've chosen these models:

**RestNet50:**

**InceptionV3:**

**Inception RestNet V2:**

**VGG 16:**

