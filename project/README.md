# 50 Years of Terrorism

Data Story: [https://adateamelec.github.io](https://adateamelec.github.io)

# Abstract

Terrorism is a subject largely covered in the media, and, unfortunately, we became accustomed to its presence worldwide, particularly over the last decade. Nevertheless, the problem we are facing today is not new. The source of certain conflicts dates from multiple decades, some of which are still lasting today. Our goal is to track and vizualize terrorism evolution through the past 50 years based on "[The Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd)". There are many questions we can ask ourselves about terrorism, such as "Is EU less safe nowadays ?", "Did attack mediums & reasons change over the years ?" or "Can we discriminate current/future conflictual zones ?". It would be presumptuous from us to say that we are going to solve major issues, or even predict futur attacks. However, through the exploration of the dataset, and by trying to answer those interrogations, we aim to grasp an overview and a better understanding to the evolution of terrorism.

# Research questions

### Data Analysis & Vizualization

- Have terrorist attack become more frequent over the years ?
- Did the number of casualties increased region-wise/worldwide, over the years ?
- Where and when did the deadliest attacks occur ?
- What is the evolution of attack's mediums & reasons ? Who is targeted ?
- Is the religious factor always responsible for terrorist attacks ? (enough data ?)

### Main Questions
- Defined conflictual areas based on the localization of the attacks.
- Did dangerous areas changed over the years. Is EU less safe nowadays ? 
- Are political context of a country and recurrence of terrorist attack linked somehow ? This question can be addressed for specific countries, e.g. Iraq or Libya.
- Are the threats in specific countries the same as in the past ?

# Dataset
The dataset is composed of a list of terrorist attacks that occurred since 1970. It is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland. It contains a numerous amount of features for each event such as localization (latitude, longitude, country), attack type, summary, number of death, number of wounded, target type, motive, claiming entity, ransom, etc..
Complete informations about the dataset on the [Global Terrorism Database](http://start.umd.edu/gtd/about/) website.


# A list of internal milestones up until project milestone 2

## 1. Handling/Cleaning  data - 15.11.17
As a first stage we will focus on looking at the dataset and check the availability and consistency of wanted fields. For example: Are all fields complete (NaN values), can we rely on them ? We will focus on building a clean data frame with date, geographical data, # casuality (injured), targeted people, weapons and motives. Note that we are looking for thoses information since they are relevant to the questions we are interested in.

## 2. Vizualization data - 25.11.17
Now that our data are cleaned we can start answering the questions mentioned above. We will plot a worldwide map with time evolution to look at the terrosist attacks over time. Also display the deadliest attacks in Europe as a map to have a better insight of the data. Plot histograms of the evolution of attack's mediums, reasons, targets and deadliest countries.

## 3. Comments - 28.11.17
Comment data and analyse the results. Have a structured plan for what comes next (further analysis and final report).

# A list of internal milestones up until project milestone 3 

## 1. Group connections graph
In our data we have 3 entries for groupe names. It is linked to the fact that some terrorist attacks are organized by multiple groups. Our data even tell us if two groups are competiting and claiming a specific attacks. We will use thoses informations to create a graph to link groups together. In other words, here we want to focus on how terrorists work together (or on their own) in certain zones using graphs. We hope to highlight cluster of groups working togther in specific regions.

## 2. Group territory evolution
An other why of looking at terrorism it to look at how group territories are evolving over the years. Al Qaida was present in Iraq few years ago. Now it is ISIS that is ruling over the same territory. We will create a graph to represent the atual world and use graph signal processing (GSP) to look at the evolution of territories over time. More precisely we will use [heat kernels](https://en.wikipedia.org/wiki/Heat_kernel) to estimate the zones of influence of each groups. This technic allows us to give importance to attacks and create intelligent representation of territories.

# Contributions
- Christian Abbet: Data reading and cleaning. Computation of groups info and location and statisitcs. Plotting of results using folium and basemap. Graph signal processing technics for groups clustering (Signed Graph) and territories location (Heat Kernels and classification)
- Nicolas Masserey: Data cleaning and processing, map generation using folium, results analysis, writting data story 
- Alexandre Poussard: Problem formulation, data processing, analysis/graph plots and story telling developpement
