# JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning

# JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning

<table>
  <tr>
    <td>
      <img src=".\images\1994-01-25.png" width="1000" alt="Brahmaputra-Jamuna River">
    </td>
    <td>
      <p style="font-size: 16px;">
        This repository stores the data, code, and other files necessary for the completion of the group project Morph 3 from the course CEGM2003 Data Science and Artificial Intelligence for engineers conducted by >Gilles Douwes, Maarten de Nooijer, Berend Bouvy, Wouter Niessen and Shijie Hu</a>, students of the MSc Applied Earth Sciences program. This group project and the repository is mainly based on the Master's thesis of <a href="https://nl.linkedin.com/in/antonio-magherini-4349b2229">Antonio Magherini</a>, student of the MSc Civil Engineering program - Hydraulic Engineering track, with a specialisation in River Engineering at the <a href="https://www.tudelft.nl/citg">Faculty of Civil Engineering and Geosciences</a> of Delft University of Technology (TU Delft).
      </p>
      <p style="font-size: 16px;">
        The manuscript can be found at <a href="https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348">TU Delft repository</a>.
      </p>
      <p style="font-size: 16px;">
        For any information, feel free to contact the author at: <a href="mailto:nooijermm@gmail.com"><em>noooijermm@gmail.com</em></a>.
      </p>
      <p style="margin-top: 100px;">
        <em>The image represents the Brahmaputra-Jamuna River at the border between India and Bangladesh. The image was taken on January 25, 1994. It was retrieved from <a href="https://earthengine.google.com/">Google Earth Engine</a> <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2">USGS Landsat 5 collection</a>.</em>
      </p>
    </td>
  </tr>
</table>

---

## Repository structure

The structure of this repository is the following:
- <code>data</code>, empty folder, data can be send if requested; 
- <code>images</code>, contains the poster presented during the final presentation; 
- <code>model</code>, contains the modules and noteboooks with the deep-learning model;
- <code>postprocessing</code>, contains the modules used for the data postprocessing;
- <code>preliminary</code>, contains the notebooks with the preliminary data analysis, satellite image visualization, preprocessing steps, and other examples; 
- <code>preprocessing</code>, contains the modules used for the data preprocessing.

---

## Install dependencies

<code>braided.yml</code> is the environment file with all dependencies, needed to run the notebooks.

To activate the environment follow these steps:

- make sure to have the file <code>braided.yml</code> in your system (for Windows users, store it in <code>C:\Windows\System32</code>);
- open the anaconda prompt;
- run <code>conda env create -f braided.yml</code>;
- verify that the environment is correctly installed by running <code>conda env list</code> and checking the environment exists;
- activate the environment by running <code>conda activate braided</code>;
- deactivate the environment by running <code>conda deactivate</code>;

---

## Cite

Please cite the [Master thesis](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348) as:

```
@mastersthesis{magherini2024,
author = {Magherini, A.},
title = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
school = {{Delft University of Technology}},
year = {2024},
month = {10},
howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
```
