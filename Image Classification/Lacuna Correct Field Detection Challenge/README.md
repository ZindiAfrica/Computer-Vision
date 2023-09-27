
# Competition Summary

## Description

Crop yield prediction is a valuable tool for agronomists and policymakers.

It is also a hard task, especially when dealing with small fields in a subsistence setting.

One challenge with many existing datasets is that of location accuracy.

GPS locations for fields can end up offset from the true location due to sensor inaccuracies or from locations being collected at the edges of fields rather than the field centres.

The objective of this competition is to design a method that can help correct these location offsets by finding the most probable field center given an input location.


## Competition Rules

Participation in this competition could be as an individual or in a team of up to four people.

Prizes are transferred only to the individual players or to the team leader.

Code was not shared privately outside of a team. Any code shared, was made available to all competition participants through the platform. (i.e. on the discussion boards).


## Datasets and packages

The model may use any publically available data (subject to approval), including any datasets that can be accessed through tools such as Google Earth Engine.

The data for this competition is part of a larger dataset of maize yields collected from East Africa.

Several data sources were provided including:

Planet Lab images: These are images captured using Planet Lab Satellite in ~4.7 m resolution in different timestamps.

Sentinel-2 images: These are images captured using Sentinel-2 Satellite in ~10 m resolution in different timestamps.

Meta-data: These are variables captured during the yield estimation process in the field which includes:

yield estimate (kgs/m^2)
field area
year in which the yield estimation process was done
annotation quality (available only in training and auxiliary data)


## Submissions and winning

The top 3 solution placed on the final leaderboard were required to submit their winning solution code to us for verification, and thereby agreed to assign all worldwide rights of copyright in and to such winning solution to Zindi.


## Reproducibility

The full documentation was retrieved. This includes:
- All data used

- Output data and where they are stored

- Explanation of features used

- The solution must include the original data provided by Zindi and validated external data (no processed data)

- All editing of data must be done in a notebook (i.e. not manually in Excel)


## Data standards:

- The most recent versions of packages were used.

- Submitted code run on the original train, test, and other datasets provided.


## Evaluation:

The evaluation metric for this competition is Mean Absolute Error, measured in kilometers.


## Prizes

1st place overall from any country: $4 000 USD.
1st place African citizen currently residing in Africa: $2 000 USD.
The 1st place female-identified African citizen currently residing in Africa: $2 000 USD.
2nd place overall from any country: $1 500 USD.
3rd place overall from any country: $500 USD.

The 'overall' prizes went to the highest placed winners. The African citizen prize was selected second. The female-identified African prize was selected third;
If a team is composed of both (female) African and non-African data scientists, then your team's category was determined by the majority in the team. If the team has a 50/50 split it will go toward female African first.


## Benefits

The solutions will be incorporated into a research project that aims to correct location errors in this dataset to produce a new high-accuracy plot location and yield dataset that can be used to better understand the agricultural landscape. 

These solutions, combined with the results of the previous Crop Yield Prediction challenge, will hopefully enable yield prediction at a higher accuracy than previously achieved.


[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg




