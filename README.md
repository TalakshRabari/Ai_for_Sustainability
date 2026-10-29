
# AI for Sustainability: Land Cover Classification in Delhi-NCR 🌍

An AI/ML-based project focused on classifying land cover across the Delhi National Capital Region (NCR) into five categories: **Cropland, Built-up, Vegetation, Water, and Others**.

The project explores how machine learning and geospatial data can support environmental monitoring, sustainable urban planning, and a better understanding of land distribution across Delhi-NCR.

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Land Cover Classes](#-land-cover-classes)
- [Study Area](#-study-area)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [Applications](#-applications)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🌱 Overview

Land cover classification is the process of identifying and categorizing the physical materials present on the Earth's surface. It helps distinguish areas occupied by agriculture, buildings, vegetation, water bodies, and other land cover types.

Delhi-NCR is a region experiencing urban development, changes in land utilization, and increasing environmental pressures. Understanding its land cover distribution can help support data-driven sustainability initiatives.

This project applies AI and machine learning techniques to classify land across Delhi-NCR into five major categories. The resulting classifications can provide a foundation for environmental analysis and future sustainability-related applications.

## 🎯 Objectives

- Develop a land cover classification workflow for Delhi-NCR.
- Categorize land into five meaningful classes.
- Explore the use of AI and machine learning for geospatial and environmental analysis.
- Support the understanding of agricultural land, built-up areas, vegetation, and water distribution.
- Establish a foundation for future land cover monitoring and sustainability research.

## 🗺️ Land Cover Classes

The project considers the following five categories:

| Class | Description |
|---|---|
| 🌾 Cropland | Areas primarily used for agricultural activities and crop cultivation. |
| 🏙️ Built-up | Urban settlements, buildings, roads, and other constructed surfaces. |
| 🌳 Vegetation | Vegetated areas, including green cover and other plant-covered land. |
| 💧 Water | Water bodies such as rivers, lakes, ponds, and reservoirs. |
| 🌐 Others | Land cover that does not fall into the four primary categories. |

The exact class definitions depend on the project's labeling scheme and input data.

## 📍 Study Area

**Delhi National Capital Region (NCR), India**

The repository includes GeoJSON files representing the geographical boundaries used for the project.

- `delhi_ncr_region.geojson`: Delhi-NCR regional boundary data.
- `delhi_airshed.geojson`: Delhi airshed boundary data.

These files can support geospatial operations such as defining the study area, spatial filtering, and visualizing the region.

## 📂 Project Structure

The repository currently contains the following main components:

```text
Ai_for_Sustainability/
├── scripts/                  # Project scripts
├── delhi_airshed.geojson     # Airshed boundary data
├── delhi_ncr_region.geojson  # Delhi-NCR boundary data
└── README.md                 # Project documentation
```

The `scripts/` directory contains the project's implementation code. Refer to the individual scripts for details about the implemented workflow, input requirements, and execution steps.

## ⚙️ Methodology

The project focuses on land cover classification using AI and machine learning. A typical workflow for this task consists of the following stages:

1. **Study Area Definition:** Define the geographical region using the available GeoJSON boundary files.
2. **Data Preparation:** Prepare the relevant land cover data and input features for classification.
3. **Preprocessing:** Process the input data into a format suitable for the classification pipeline.
4. **Model Development:** Apply the machine learning approach implemented in the project to distinguish between the five land cover classes.
5. **Classification:** Assign the appropriate land cover category to the input observations or geographical areas.
6. **Analysis and Visualization:** Examine the resulting class distribution and visualize the spatial patterns where supported by the implementation.

The exact data sources, preprocessing operations, model architecture, and evaluation methods should be documented according to the implementation in the repository.

## 🛠️ Technology Stack

The project is based on the following broad technologies:

- **Programming Language:** Python
- **Core Approach:** Artificial Intelligence and Machine Learning
- **Application Domain:** Geospatial Analysis and Environmental Sustainability
- **Geospatial Data Format:** GeoJSON

Additional libraries and frameworks should be listed once the dependencies used by the project scripts are confirmed.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TalakshRabari/Ai_for_Sustainability.git
```

### 2. Navigate to the Project Directory

```bash
cd Ai_for_Sustainability
```

### 3. Set Up Python

Create and activate a virtual environment if your implementation requires one.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install the Python packages required by the project scripts. Refer to the dependency information in the repository, if provided.

For example, if the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Run the Project

Review the scripts inside the `scripts/` directory to identify the appropriate entry point, required input data, and execution command.

The exact run command depends on the project's implementation.

## 🌏 Applications

Land cover classification can contribute to several sustainability-related applications:

- **Urban Planning:** Understand the distribution of built-up areas and surrounding land.
- **Agricultural Monitoring:** Identify areas classified as cropland.
- **Green Cover Assessment:** Analyze the spatial distribution of vegetation.
- **Water Resource Planning:** Identify mapped water-covered areas.
- **Environmental Monitoring:** Support land cover analysis for sustainability studies.
- **Data-Driven Decision Making:** Provide classified land cover information for further geographical and environmental analysis.

These are potential applications of the classification outputs, rather than claims of features already implemented in the project.

## 🔮 Future Improvements

Potential directions for extending the project include:

- Visualizing classified land cover on an interactive map.
- Evaluating model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Comparing different machine learning and deep learning models.
- Generating class-wise land cover statistics for Delhi-NCR.
- Comparing land cover classifications across different time periods, when suitable data is available.
- Investigating changes in built-up areas, cropland, vegetation, and water coverage.
- Developing a dashboard to make classification outputs easier to explore.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a new branch for your changes.
3. Implement and test your improvements.
4. Submit a pull request with a description of your changes.

## 📄 License

No license information is specified here. Please add a license file and update this section if you intend to distribute the project for reuse.

## 👨‍💻 Author

**Talaksh Rabari**

- GitHub: [@TalakshRabari](https://github.com/TalakshRabari)
- Project Repository: [AI for Sustainability](https://github.com/TalakshRabari/Ai_for_Sustainability)

---

*Using AI and geospatial analysis to support a more sustainable understanding of land cover across Delhi-NCR.*
