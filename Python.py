{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOz5/H9Qq9IxOQOJJuLlQqs",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sleepygirlleen/Machine-Learning-Terapan/blob/main/Python.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#**Proyek Machine Learning Terapan 1**\n",
        "\n",
        "- **Nama:** Sulistiani\n",
        "- **Email:** lisasa2lilisa@gmail.com\n",
        "- **ID Dicoding:** hi_itslizeu"
      ],
      "metadata": {
        "id": "1UvPAyor2Zdm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Import Library**"
      ],
      "metadata": {
        "id": "a8yyBVaR3b-x"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "aBt5m29z2U70"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "import numpy\n",
        "from sklearn import metrics\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "%matplotlib inline\n",
        "from IPython.display import display\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from typing import Dict, List, Tuple\n",
        "from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer\n",
        "from sklearn.base import BaseEstimator\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.svm import SVR\n",
        "from sklearn.neighbors import KNeighborsRegressor"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Import Dataset**"
      ],
      "metadata": {
        "id": "Xe76SmZ43qJX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Import dataset\n",
        "df = pd.read_csv('student_habits_performance.csv')"
      ],
      "metadata": {
        "id": "3DIkMCYc3pdC"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Assessing Data**"
      ],
      "metadata": {
        "id": "zScTtTa_4Qhf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Menampilkan ukuran dataset\n",
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r3jKO_ws4L2X",
        "outputId": "7aae9c20-91ba-425c-dab6-a75f524046e1"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1000, 16)"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Menampilkan dataset\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "CFLsG7lq4cAf",
        "outputId": "6de99e24-f7b0-47ae-e6cf-a9bf5b3f02fb"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  student_id  age  gender  study_hours_per_day  social_media_hours  \\\n",
              "0      S1000   23  Female                  0.0                 1.2   \n",
              "1      S1001   20  Female                  6.9                 2.8   \n",
              "2      S1002   21    Male                  1.4                 3.1   \n",
              "3      S1003   23  Female                  1.0                 3.9   \n",
              "4      S1004   19  Female                  5.0                 4.4   \n",
              "\n",
              "   netflix_hours part_time_job  attendance_percentage  sleep_hours  \\\n",
              "0            1.1            No                   85.0          8.0   \n",
              "1            2.3            No                   97.3          4.6   \n",
              "2            1.3            No                   94.8          8.0   \n",
              "3            1.0            No                   71.0          9.2   \n",
              "4            0.5            No                   90.9          4.9   \n",
              "\n",
              "  diet_quality  exercise_frequency parental_education_level internet_quality  \\\n",
              "0         Fair                   6                   Master          Average   \n",
              "1         Good                   6              High School          Average   \n",
              "2         Poor                   1              High School             Poor   \n",
              "3         Poor                   4                   Master             Good   \n",
              "4         Fair                   3                   Master             Good   \n",
              "\n",
              "   mental_health_rating extracurricular_participation  exam_score  \n",
              "0                     8                           Yes        56.2  \n",
              "1                     8                            No       100.0  \n",
              "2                     1                            No        34.3  \n",
              "3                     1                           Yes        26.8  \n",
              "4                     1                            No        66.4  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e0614d8e-9e27-4070-bc8a-561d6f3ce043\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>student_id</th>\n",
              "      <th>age</th>\n",
              "      <th>gender</th>\n",
              "      <th>study_hours_per_day</th>\n",
              "      <th>social_media_hours</th>\n",
              "      <th>netflix_hours</th>\n",
              "      <th>part_time_job</th>\n",
              "      <th>attendance_percentage</th>\n",
              "      <th>sleep_hours</th>\n",
              "      <th>diet_quality</th>\n",
              "      <th>exercise_frequency</th>\n",
              "      <th>parental_education_level</th>\n",
              "      <th>internet_quality</th>\n",
              "      <th>mental_health_rating</th>\n",
              "      <th>extracurricular_participation</th>\n",
              "      <th>exam_score</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>S1000</td>\n",
              "      <td>23</td>\n",
              "      <td>Female</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.2</td>\n",
              "      <td>1.1</td>\n",
              "      <td>No</td>\n",
              "      <td>85.0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>Fair</td>\n",
              "      <td>6</td>\n",
              "      <td>Master</td>\n",
              "      <td>Average</td>\n",
              "      <td>8</td>\n",
              "      <td>Yes</td>\n",
              "      <td>56.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>S1001</td>\n",
              "      <td>20</td>\n",
              "      <td>Female</td>\n",
              "      <td>6.9</td>\n",
              "      <td>2.8</td>\n",
              "      <td>2.3</td>\n",
              "      <td>No</td>\n",
              "      <td>97.3</td>\n",
              "      <td>4.6</td>\n",
              "      <td>Good</td>\n",
              "      <td>6</td>\n",
              "      <td>High School</td>\n",
              "      <td>Average</td>\n",
              "      <td>8</td>\n",
              "      <td>No</td>\n",
              "      <td>100.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>S1002</td>\n",
              "      <td>21</td>\n",
              "      <td>Male</td>\n",
              "      <td>1.4</td>\n",
              "      <td>3.1</td>\n",
              "      <td>1.3</td>\n",
              "      <td>No</td>\n",
              "      <td>94.8</td>\n",
              "      <td>8.0</td>\n",
              "      <td>Poor</td>\n",
              "      <td>1</td>\n",
              "      <td>High School</td>\n",
              "      <td>Poor</td>\n",
              "      <td>1</td>\n",
              "      <td>No</td>\n",
              "      <td>34.3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>S1003</td>\n",
              "      <td>23</td>\n",
              "      <td>Female</td>\n",
              "      <td>1.0</td>\n",
              "      <td>3.9</td>\n",
              "      <td>1.0</td>\n",
              "      <td>No</td>\n",
              "      <td>71.0</td>\n",
              "      <td>9.2</td>\n",
              "      <td>Poor</td>\n",
              "      <td>4</td>\n",
              "      <td>Master</td>\n",
              "      <td>Good</td>\n",
              "      <td>1</td>\n",
              "      <td>Yes</td>\n",
              "      <td>26.8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>S1004</td>\n",
              "      <td>19</td>\n",
              "      <td>Female</td>\n",
              "      <td>5.0</td>\n",
              "      <td>4.4</td>\n",
              "      <td>0.5</td>\n",
              "      <td>No</td>\n",
              "      <td>90.9</td>\n",
              "      <td>4.9</td>\n",
              "      <td>Fair</td>\n",
              "      <td>3</td>\n",
              "      <td>Master</td>\n",
              "      <td>Good</td>\n",
              "      <td>1</td>\n",
              "      <td>No</td>\n",
              "      <td>66.4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e0614d8e-9e27-4070-bc8a-561d6f3ce043')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e0614d8e-9e27-4070-bc8a-561d6f3ce043 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e0614d8e-9e27-4070-bc8a-561d6f3ce043');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-22a2a8c9-25c8-4ccc-ad81-49df4cfc5155\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-22a2a8c9-25c8-4ccc-ad81-49df4cfc5155')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-22a2a8c9-25c8-4ccc-ad81-49df4cfc5155 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1000,\n  \"fields\": [\n    {\n      \"column\": \"student_id\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 1000,\n        \"samples\": [\n          \"S1521\",\n          \"S1737\",\n          \"S1740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 17,\n        \"max\": 24,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          20,\n          18,\n          23\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"gender\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Female\",\n          \"Male\",\n          \"Other\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"study_hours_per_day\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.4688899303990155,\n        \"min\": 0.0,\n        \"max\": 8.3,\n        \"num_unique_values\": 78,\n        \"samples\": [\n          5.4,\n          0.0,\n          2.2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"social_media_hours\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.1724224171877315,\n        \"min\": 0.0,\n        \"max\": 7.2,\n        \"num_unique_values\": 60,\n        \"samples\": [\n          1.2,\n          1.3,\n          0.1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"netflix_hours\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.0751175692861612,\n        \"min\": 0.0,\n        \"max\": 5.4,\n        \"num_unique_values\": 51,\n        \"samples\": [\n          0.3,\n          3.7,\n          4.6\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"part_time_job\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"Yes\",\n          \"No\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"attendance_percentage\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.399246296429354,\n        \"min\": 56.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 320,\n        \"samples\": [\n          75.7,\n          90.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sleep_hours\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.226376773259339,\n        \"min\": 3.2,\n        \"max\": 10.0,\n        \"num_unique_values\": 68,\n        \"samples\": [\n          8.2,\n          6.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"diet_quality\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Fair\",\n          \"Good\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exercise_frequency\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          6,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"parental_education_level\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Master\",\n          \"High School\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"internet_quality\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Average\",\n          \"Poor\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"mental_health_rating\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 1,\n        \"max\": 10,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          2,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"extracurricular_participation\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"No\",\n          \"Yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exam_score\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 16.88856392181825,\n        \"min\": 18.4,\n        \"max\": 100.0,\n        \"num_unique_values\": 480,\n        \"samples\": [\n          53.5,\n          59.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Informasi Dataset\n",
        "\n",
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HNDrmkaI4nmA",
        "outputId": "df8de2f7-5bd3-49e7-9b7c-b70b65760640"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1000 entries, 0 to 999\n",
            "Data columns (total 16 columns):\n",
            " #   Column                         Non-Null Count  Dtype  \n",
            "---  ------                         --------------  -----  \n",
            " 0   student_id                     1000 non-null   object \n",
            " 1   age                            1000 non-null   int64  \n",
            " 2   gender                         1000 non-null   object \n",
            " 3   study_hours_per_day            1000 non-null   float64\n",
            " 4   social_media_hours             1000 non-null   float64\n",
            " 5   netflix_hours                  1000 non-null   float64\n",
            " 6   part_time_job                  1000 non-null   object \n",
            " 7   attendance_percentage          1000 non-null   float64\n",
            " 8   sleep_hours                    1000 non-null   float64\n",
            " 9   diet_quality                   1000 non-null   object \n",
            " 10  exercise_frequency             1000 non-null   int64  \n",
            " 11  parental_education_level       909 non-null    object \n",
            " 12  internet_quality               1000 non-null   object \n",
            " 13  mental_health_rating           1000 non-null   int64  \n",
            " 14  extracurricular_participation  1000 non-null   object \n",
            " 15  exam_score                     1000 non-null   float64\n",
            "dtypes: float64(6), int64(3), object(7)\n",
            "memory usage: 125.1+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Checking Missing Values**"
      ],
      "metadata": {
        "id": "NQC6k5U24v12"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 586
        },
        "id": "d1BU15rF43eu",
        "outputId": "273e4056-bf16-47ee-dbb4-bcba11fdbcc2"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "student_id                        0\n",
              "age                               0\n",
              "gender                            0\n",
              "study_hours_per_day               0\n",
              "social_media_hours                0\n",
              "netflix_hours                     0\n",
              "part_time_job                     0\n",
              "attendance_percentage             0\n",
              "sleep_hours                       0\n",
              "diet_quality                      0\n",
              "exercise_frequency                0\n",
              "parental_education_level         91\n",
              "internet_quality                  0\n",
              "mental_health_rating              0\n",
              "extracurricular_participation     0\n",
              "exam_score                        0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>student_id</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>gender</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>study_hours_per_day</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>social_media_hours</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>netflix_hours</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>part_time_job</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>attendance_percentage</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>sleep_hours</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>diet_quality</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>exercise_frequency</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>parental_education_level</th>\n",
              "      <td>91</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>internet_quality</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mental_health_rating</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>extracurricular_participation</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>exam_score</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Cleaning Data**"
      ],
      "metadata": {
        "id": "WRPExFbV5YfG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Hapus kolom yang tidak diperlukan\n",
        "\n",
        "df.drop(columns=['student_id', 'social_media_hours', 'netflix_hours', 'gender', 'part_time_job', 'parental_education_level', 'extracurricular_participation', 'internet_quality'], inplace=True)"
      ],
      "metadata": {
        "id": "a1_IoYs35bu2"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "emuUTuR3-L2i",
        "outputId": "644a4884-8939-46bf-974d-c0904242cb3c"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age  study_hours_per_day  attendance_percentage  sleep_hours diet_quality  \\\n",
              "0   23                  0.0                   85.0          8.0         Fair   \n",
              "1   20                  6.9                   97.3          4.6         Good   \n",
              "2   21                  1.4                   94.8          8.0         Poor   \n",
              "3   23                  1.0                   71.0          9.2         Poor   \n",
              "4   19                  5.0                   90.9          4.9         Fair   \n",
              "\n",
              "   exercise_frequency  mental_health_rating  exam_score  \n",
              "0                   6                     8        56.2  \n",
              "1                   6                     8       100.0  \n",
              "2                   1                     1        34.3  \n",
              "3                   4                     1        26.8  \n",
              "4                   3                     1        66.4  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ff27749d-e637-4a4c-9e68-815292725f12\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>study_hours_per_day</th>\n",
              "      <th>attendance_percentage</th>\n",
              "      <th>sleep_hours</th>\n",
              "      <th>diet_quality</th>\n",
              "      <th>exercise_frequency</th>\n",
              "      <th>mental_health_rating</th>\n",
              "      <th>exam_score</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>23</td>\n",
              "      <td>0.0</td>\n",
              "      <td>85.0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>Fair</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>56.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20</td>\n",
              "      <td>6.9</td>\n",
              "      <td>97.3</td>\n",
              "      <td>4.6</td>\n",
              "      <td>Good</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>100.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>21</td>\n",
              "      <td>1.4</td>\n",
              "      <td>94.8</td>\n",
              "      <td>8.0</td>\n",
              "      <td>Poor</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>34.3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>23</td>\n",
              "      <td>1.0</td>\n",
              "      <td>71.0</td>\n",
              "      <td>9.2</td>\n",
              "      <td>Poor</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>26.8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>19</td>\n",
              "      <td>5.0</td>\n",
              "      <td>90.9</td>\n",
              "      <td>4.9</td>\n",
              "      <td>Fair</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>66.4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ff27749d-e637-4a4c-9e68-815292725f12')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-ff27749d-e637-4a4c-9e68-815292725f12 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-ff27749d-e637-4a4c-9e68-815292725f12');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-d8fdd7aa-64cf-466c-8419-61ee5ebec1fd\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-d8fdd7aa-64cf-466c-8419-61ee5ebec1fd')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-d8fdd7aa-64cf-466c-8419-61ee5ebec1fd button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1000,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 17,\n        \"max\": 24,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          20,\n          18,\n          23\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"study_hours_per_day\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.4688899303990155,\n        \"min\": 0.0,\n        \"max\": 8.3,\n        \"num_unique_values\": 78,\n        \"samples\": [\n          5.4,\n          0.0,\n          2.2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"attendance_percentage\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.399246296429354,\n        \"min\": 56.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 320,\n        \"samples\": [\n          75.7,\n          90.3,\n          74.7\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sleep_hours\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.226376773259339,\n        \"min\": 3.2,\n        \"max\": 10.0,\n        \"num_unique_values\": 68,\n        \"samples\": [\n          8.2,\n          6.0,\n          7.4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"diet_quality\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Fair\",\n          \"Good\",\n          \"Poor\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exercise_frequency\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          6,\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"mental_health_rating\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 1,\n        \"max\": 10,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          2,\n          1,\n          9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exam_score\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 16.88856392181825,\n        \"min\": 18.4,\n        \"max\": 100.0,\n        \"num_unique_values\": 480,\n        \"samples\": [\n          53.5,\n          59.3,\n          88.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "wobyJOuwoWCN",
        "outputId": "ced598e4-189b-4347-e05f-2af788eeb469"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "age                      0\n",
              "study_hours_per_day      0\n",
              "attendance_percentage    0\n",
              "sleep_hours              0\n",
              "diet_quality             0\n",
              "exercise_frequency       0\n",
              "mental_health_rating     0\n",
              "exam_score               0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>study_hours_per_day</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>attendance_percentage</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>sleep_hours</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>diet_quality</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>exercise_frequency</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mental_health_rating</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>exam_score</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.duplicated().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aso4zatnossn",
        "outputId": "3bdc4ad0-ba2c-4901-96cb-9c01bd574e5f"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.int64(0)"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "numeric_features = df.select_dtypes(include=np.number).columns.tolist()\n",
        "\n",
        "for feature in numeric_features:\n",
        "    plt.figure(figsize=(10, 6))\n",
        "    sns.boxplot(x=df[feature])\n",
        "    plt.title(f'Box Plot of {feature}')\n",
        "    plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "9voGBGD_Jx7V",
        "outputId": "86506d42-da66-4696-93cf-efea8afa151d"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAJTJJREFUeJzt3XmQVfWZ+P+nm6VphW5EAW3Z1ERwR5EkqMSlFEVkNNFYokmAIXGqRBw05SQGxxA141hqjOM4JhgFhSTjkIlbjOBGNC6pcgGXCSIqamRpI8oqW+jz/SM/7i8tsojy3Kb79arq0j733NtP84Hmvu+551BRFEURAAAA21lluQcAAABaBvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QFAvPnmm1FRURGTJk0q9yiNTJs2Lfr27Rvt2rWLioqKWLJkSblHAuBTEB8An6FJkyZFRUVFo48uXbrEscceGw888ED6PL///e8bzdKmTZvYe++945vf/Ga88cYbn8nXeOqpp2L8+PGfeRgsXrw4zjzzzKiuro6bbropJk+eHDvvvPNn+jUAyNW63AMANEeXX3557LXXXlEURdTX18ekSZPi5JNPjvvuuy9OOeWU9HkuuOCC6N+/f6xbty6ef/75mDBhQtx///3x0ksvRV1d3ad67Keeeip++MMfxogRI6Jjx46fzcAR8cwzz8Ty5cvjiiuuiOOPP/4ze1wAykd8AGwHgwcPjsMPP7z0+ahRo6Jr167xq1/9qizxMXDgwDjjjDMiImLkyJGx7777xgUXXBC33357XHLJJenzbI133303IuIzDRoAysvbrgASdOzYMaqrq6N168av+axcuTK+853vRPfu3aOqqip69+4d1157bRRFERERq1atij59+kSfPn1i1apVpfu9//77sccee8QRRxwR69ev/8TzHHfccRERMW/evM3u9+ijj8bAgQNj5513jo4dO8app54as2fPLt0+fvz4uPjiiyMiYq+99iq9vevNN9/c7ONOnTo1+vXrF9XV1bHbbrvF17/+9Zg/f37p9mOOOSaGDx8eERH9+/ePioqKGDFixCYf76233orzzjsvevfuHdXV1bHrrrvG1772tY+d48UXX4yjjz46qquro1u3bnHllVfGxIkTP3buBx54oPT9d+jQIYYMGRL/93//t9nvDYBNc+QDYDtYunRpvPfee1EURbz77rtx4403xooVK+LrX/96aZ+iKOIf/uEfYsaMGTFq1Kjo27dvTJ8+PS6++OKYP39+XH/99VFdXR233357HHnkkTFu3Lj48Y9/HBERo0ePjqVLl8akSZOiVatWn3i+119/PSIidt11103u8/DDD8fgwYNj7733jvHjx8eqVavixhtvjCOPPDKef/756NWrV3z1q1+NV199NX71q1/F9ddfH7vttltERHTu3HmTjztp0qQYOXJk9O/fP6666qqor6+PG264IZ588smYOXNmdOzYMcaNGxe9e/eOCRMmlN7Cts8++2zyMZ955pl46qmn4qyzzopu3brFm2++GTfffHMcc8wx8ac//Sl22mmniIiYP39+HHvssVFRURGXXHJJ7LzzzvHzn/88qqqqNnrMyZMnx/Dhw+PEE0+Mq6++Oj788MO4+eab46ijjoqZM2dGr169tuaXGoC/VwDwmZk4cWIRERt9VFVVFZMmTWq07913311ERHHllVc22n7GGWcUFRUVxWuvvVbadskllxSVlZXF448/XkydOrWIiOInP/nJFueZMWNGERHFbbfdVvzlL38pFixYUNx///1Fr169ioqKiuKZZ54piqIo5s2bV0REMXHixNJ9+/btW3Tp0qVYvHhxadsLL7xQVFZWFt/85jdL26655poiIop58+ZtcZ61a9cWXbp0KQ488MBi1apVpe2//e1vi4goLrvsstK2Db+WG2bcnA8//HCjbU8//XQREcUdd9xR2jZmzJiioqKimDlzZmnb4sWLi06dOjX6HpYvX1507Nix+Pa3v93oMRctWlTU1tZutB2AreNtVwDbwU033RQPPfRQPPTQQzFlypQ49thj41vf+lb85je/Ke3zu9/9Llq1ahUXXHBBo/t+5zvfiaIoGl0da/z48XHAAQfE8OHD47zzzoujjz56o/ttzj/+4z9G586do66uLoYMGRIrV66M22+/vdF5KX9v4cKFMWvWrBgxYkR06tSptP3ggw+OE044IX73u99t9df+e88++2y8++67cd5550W7du1K24cMGRJ9+vSJ+++/f5set7q6uvT/69ati8WLF8fnPve56NixYzz//POl26ZNmxYDBgyIvn37lrZ16tQpzjnnnEaP99BDD8WSJUti2LBh8d5775U+WrVqFV/84hdjxowZ2zQnQEvnbVcA28EXvvCFRk/shw0bFoceemicf/75ccopp0Tbtm3jrbfeirq6uujQoUOj++63334R8bfzGDZo27Zt3HbbbdG/f/9o165d6RyFrXXZZZfFwIEDo1WrVrHbbrvFfvvtt9H5J39vw9fu3bv3Rrftt99+MX369Fi5cuUnvvTt5h63T58+8cQTT3yix9tg1apVcdVVV8XEiRNj/vz5pXNmIv72Fri///oDBgzY6P6f+9znGn0+d+7ciPj/z435qJqamm2aE6ClEx8ACSorK+PYY4+NG264IebOnRsHHHDAJ36M6dOnR0TE6tWrY+7cubHXXntt9X0POuigZn252jFjxsTEiRNj7NixMWDAgKitrY2Kioo466yzoqGh4RM/3ob7TJ48OXbfffeNbt9cuAGwaX56AiT561//GhERK1asiIiInj17xsMPPxzLly9vdPTjlVdeKd2+wYsvvhiXX355jBw5MmbNmhXf+ta34qWXXora2trtMuuGrz1nzpyNbnvllVdit912Kx31+CRHYP7+cT96VGHOnDmNvudP4te//nUMHz48rrvuutK21atXb/QPH/bs2TNee+21je7/0W0bTm7v0qVLs442gGzO+QBIsG7dunjwwQejbdu2pbdVnXzyybF+/fr4z//8z0b7Xn/99VFRURGDBw8u3XfEiBFRV1cXN9xwQ0yaNCnq6+vjwgsv3G7z7rHHHtG3b9+4/fbbGz2Bf/nll+PBBx+Mk08+ubRtQ4Rszb9wfvjhh0eXLl3ipz/9aaxZs6a0/YEHHojZs2fHkCFDtmneVq1aNXqrVUTEjTfeuNFliE888cR4+umnY9asWaVt77//fvziF7/YaL+ampr4t3/7t1i3bt1GX+8vf/nLNs0J0NI58gGwHTzwwAOlIxjvvvtu/PKXv4y5c+fG9773vdL5AkOHDo1jjz02xo0bF2+++WYccsgh8eCDD8Y999wTY8eOLb36fuWVV8asWbPikUceiQ4dOsTBBx8cl112WVx66aVxxhlnNAqBz9I111wTgwcPjgEDBsSoUaNKl9qtra2N8ePHl/br169fRESMGzcuzjrrrGjTpk0MHTr0Y88HadOmTVx99dUxcuTIOProo2PYsGGlS+326tVrm4PqlFNOicmTJ0dtbW3sv//+8fTTT8fDDz+80aWE/+Vf/iWmTJkSJ5xwQowZM6Z0qd0ePXrE+++/XzqKU1NTEzfffHN84xvfiMMOOyzOOuus6Ny5c7z99ttx//33x5FHHrlRNAKwFcp8tS2AZuXjLrXbrl27om/fvsXNN99cNDQ0NNp/+fLlxYUXXljU1dUVbdq0KT7/+c8X11xzTWm/5557rmjdunUxZsyYRvf761//WvTv37+oq6srPvjgg03Os+FSu1OnTt3s3B93qd2iKIqHH364OPLII4vq6uqipqamGDp0aPGnP/1po/tfccUVxZ577llUVlZu1WV377zzzuLQQw8tqqqqik6dOhXnnHNO8c477zTa55NcaveDDz4oRo4cWey2225F+/btixNPPLF45ZVXip49exbDhw9vtO/MmTOLgQMHFlVVVUW3bt2Kq666qviP//iPIiKKRYsWNdp3xowZxYknnljU1tYW7dq1K/bZZ59ixIgRxbPPPrvFmQDYWEVRfOQ4NQC0MGPHjo2f/exnsWLFim36RxsB2DrO+QCgRVm1alWjzxcvXhyTJ0+Oo446SngAbGfO+QCgRRkwYEAcc8wxsd9++0V9fX3ceuutsWzZsvjXf/3Xco8G0OyJDwBalJNPPjl+/etfx4QJE6KioiIOO+ywuPXWW+PLX/5yuUcDaPac8wEAAKRwzgcAAJBCfAAAACm2+ZyPhoaGWLBgQXTo0KH0jzIBAAAtT1EUsXz58qirq4vKyk0f39jm+FiwYEF07959W+8OAAA0M3/+85+jW7dum7x9m+OjQ4cOpS9QU1OzrQ8DAADs4JYtWxbdu3cvNcKmbHN8bHirVU1NjfgAAAC2eDqGE84BAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABI0brcAwAR9fX1sXTp0nKPAQCbVVtbG127di33GOzAxAeUWX19fXz9G9+MdWvXlHsUANisNm2rYsrkOwQI20x8QJktXbo01q1dE6v2Pjoa2tWWexySVK5aEtXzHo9Ve305Gqo7lnscgC2qXL004o3HYunSpeKDbSY+oIloaFcbDTvvVu4xSNZQ3dG6A9BiOOEcAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTNIj5Wr14dr776aqxevbrcowAAQIod8Tlws4iPt99+O84999x4++23yz0KAACk2BGfAzeL+AAAAJo+8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAECK1lu745o1a2LNmjWlz5ctW7ZdBvo03nrrrXKPAJ+Y37cA7Ej8vdV07IhrsdXxcdVVV8UPf/jD7TnLp/ajH/2o3CMAADRrnm/xaWx1fFxyySVx0UUXlT5ftmxZdO/efbsMta3GjRsXPXv2LPcY8Im89dZbfpADsMPwfKvp2BGfQ2x1fFRVVUVVVdX2nOVT69mzZ+y7777lHgMAoNnyfItPwwnnAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApGgW8dGjR4+YMGFC9OjRo9yjAABAih3xOXDrcg/wWWjXrl3su+++5R4DAADS7IjPgZvFkQ8AAKDpEx8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApBAfAABACvEBAACkEB8AAEAK8QEAAKQQHwAAQArxAQAApGhd7gGAv6lcvbTcI5CoctWSRv8FaOr8PcVnQXxAmdXW1kabtlURbzxW7lEog+p5j5d7BICt1qZtVdTW1pZ7DHZg4gPKrGvXrjFl8h2xdKlXlABo2mpra6Nr167lHoMdmPiAJqBr165+mAMAzZ4TzgEAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEjRelvvWBRFREQsW7bsMxsGAADY8Wxogg2NsCnbHB/Lly+PiIju3btv60MAAADNyPLly6O2tnaTt1cUW8qTTWhoaIgFCxZEhw4doqKiYpsH/CwsW7YsunfvHn/+85+jpqamrLPwN9akabEeTY81aXqsSdNiPZoea9L0NKU1KYoili9fHnV1dVFZuekzO7b5yEdlZWV069ZtW+++XdTU1JT9F57GrEnTYj2aHmvS9FiTpsV6ND3WpOlpKmuyuSMeGzjhHAAASCE+AACAFM0iPqqqquIHP/hBVFVVlXsU/j/WpGmxHk2PNWl6rEnTYj2aHmvS9OyIa7LNJ5wDAAB8Es3iyAcAAND0iQ8AACCF+AAAAFKIDwAAIMUOFR+PP/54DB06NOrq6qKioiLuvvvuRrdXVFR87Mc111xTnoGbuS2tx4oVK+L888+Pbt26RXV1dey///7x05/+tDzDthBbWpP6+voYMWJE1NXVxU477RQnnXRSzJ07tzzDtgBXXXVV9O/fPzp06BBdunSJ0047LebMmdNon9WrV8fo0aNj1113jfbt28fpp58e9fX1ZZq4+duaNZkwYUIcc8wxUVNTExUVFbFkyZLyDNtCbGlN3n///RgzZkz07t07qquro0ePHnHBBRfE0qVLyzh187U1f0b+6Z/+KfbZZ5+orq6Ozp07x6mnnhqvvPJKmSZu/rZmTTYoiiIGDx78sc8BmoodKj5WrlwZhxxySNx0000fe/vChQsbfdx2221RUVERp59+evKkLcOW1uOiiy6KadOmxZQpU2L27NkxduzYOP/88+Pee+9NnrTl2NyaFEURp512Wrzxxhtxzz33xMyZM6Nnz55x/PHHx8qVK8swbfP32GOPxejRo+OPf/xjPPTQQ7Fu3boYNGhQo1/vCy+8MO67776YOnVqPPbYY7FgwYL46le/Wsapm7etWZMPP/wwTjrppPj+979fxklbji2tyYIFC2LBggVx7bXXxssvvxyTJk2KadOmxahRo8o8efO0NX9G+vXrFxMnTozZs2fH9OnToyiKGDRoUKxfv76MkzdfW7MmG/zkJz+JioqKMkz5CRQ7qIgo7rrrrs3uc+qppxbHHXdczkAt3MetxwEHHFBcfvnljbYddthhxbhx4xIna7k+uiZz5swpIqJ4+eWXS9vWr19fdO7cubjlllvKMGHL8+677xYRUTz22GNFURTFkiVLijZt2hRTp04t7TN79uwiIoqnn366XGO2KB9dk783Y8aMIiKKDz74IH+wFmxza7LB//zP/xRt27Yt1q1blzhZy7Q16/HCCy8UEVG89tpriZO1XJtak5kzZxZ77rlnsXDhwq16nlwuO9SRj0+ivr4+7r//fq+MlNERRxwR9957b8yfPz+KoogZM2bEq6++GoMGDSr3aC3SmjVrIiKiXbt2pW2VlZVRVVUVTzzxRLnGalE2vE2kU6dOERHx3HPPxbp16+L4448v7dOnT5/o0aNHPP3002WZsaX56JpQfluzJkuXLo2amppo3bp11lgt1pbWY+XKlTFx4sTYa6+9onv37pmjtVgftyYffvhhnH322XHTTTfF7rvvXq7RtkqzjY/bb789OnTo4O0LZXTjjTfG/vvvH926dYu2bdvGSSedFDfddFN8+ctfLvdoLdKGJ7WXXHJJfPDBB7F27dq4+uqr45133omFCxeWe7xmr6GhIcaOHRtHHnlkHHjggRERsWjRomjbtm107Nix0b5du3aNRYsWlWHKluXj1oTy2po1ee+99+KKK66Ic889N3m6lmdz6/Ff//Vf0b59+2jfvn088MAD8dBDD0Xbtm3LNGnLsak1ufDCC+OII46IU089tYzTbZ1m+5LBbbfdFuecc06jV3nJdeONN8Yf//jHuPfee6Nnz57x+OOPx+jRo6Ourq7RK73kaNOmTfzmN7+JUaNGRadOnaJVq1Zx/PHHx+DBg6MoinKP1+yNHj06Xn75ZUeZmhBr0vRsaU2WLVsWQ4YMif333z/Gjx+fO1wLtLn1OOecc+KEE06IhQsXxrXXXhtnnnlmPPnkk553bWcftyb33ntvPProozFz5swyTrb1mmV8/OEPf4g5c+bEnXfeWe5RWqxVq1bF97///bjrrrtiyJAhERFx8MEHx6xZs+Laa68VH2XSr1+/mDVrVixdujTWrl0bnTt3ji9+8Ytx+OGHl3u0Zu3888+P3/72t/H4449Ht27dStt33333WLt2bSxZsqTR0Y/6+vomf9h8R7epNaF8trQmy5cvj5NOOik6dOgQd911V7Rp06YMU7YcW1qP2traqK2tjc9//vPxpS99KXbZZZe46667YtiwYWWYtmXY1Jo8+uij8frrr290FP3000+PgQMHxu9///vcQbegWb7t6tZbb41+/frFIYccUu5RWqx169bFunXrorKy8W+xVq1aRUNDQ5mmYoPa2tro3LlzzJ07N5599tkd4jDtjqgoijj//PPjrrvuikcffTT22muvRrf369cv2rRpE4888khp25w5c+Ltt9+OAQMGZI/bImxpTci3NWuybNmyGDRoULRt2zbuvfder65vR9vyZ6QoiiiKonRuIZ+tLa3J9773vXjxxRdj1qxZpY+IiOuvvz4mTpxYhok3b4c68rFixYp47bXXSp/PmzcvZs2aFZ06dYoePXpExN9+QE2dOjWuu+66co3ZYmxpPY4++ui4+OKLo7q6Onr27BmPPfZY3HHHHfHjH/+4jFM3b1tak6lTp0bnzp2jR48e8dJLL8U///M/x2mnneYiANvJ6NGj45e//GXcc8890aFDh9J5HLW1tVFdXR21tbUxatSouOiii6JTp05RU1MTY8aMiQEDBsSXvvSlMk/fPG1pTSL+di7OokWLSn+WXnrppejQoUP06NHDienbwZbWZEN4fPjhhzFlypRYtmxZLFu2LCIiOnfuHK1atSrn+M3OltbjjTfeiDvvvDMGDRoUnTt3jnfeeSf+/d//Paqrq+Pkk08u8/TN05bWZPfdd//Yo+U9evRomi+wlO06W9tgw2UPP/oxfPjw0j4/+9nPiurq6mLJkiXlG7SF2NJ6LFy4sBgxYkRRV1dXtGvXrujdu3dx3XXXFQ0NDeUdvBnb0prccMMNRbdu3Yo2bdoUPXr0KC699NJizZo15R26Gfu4tYiIYuLEiaV9Vq1aVZx33nnFLrvsUuy0007FV77ylWLhwoXlG7qZ25o1+cEPfrDFffjsbGlNNvVzLSKKefPmlXX25mhL6zF//vxi8ODBRZcuXYo2bdoU3bp1K84+++zilVdeKe/gzdjW/Nz6uPs01UvtVhSFM00BAIDtr1me8wEAADQ94gMAAEghPgAAgBTiAwAASCE+AACAFOIDAABIIT4AAIAU4gMAAEghPgAAgBTiAwAASCE+AACAFOIDgI1MmzYtjjrqqOjYsWPsuuuuccopp8Trr79euv2pp56Kvn37Rrt27eLwww+Pu+++OyoqKmLWrFmlfV5++eUYPHhwtG/fPrp27Rrf+MY34r333ivDdwNAUyE+ANjIypUr46KLLopnn302HnnkkaisrIyvfOUr0dDQEMuWLYuhQ4fGQQcdFM8//3xcccUV8d3vfrfR/ZcsWRLHHXdcHHroofHss8/GtGnTor6+Ps4888wyfUcANAUVRVEU5R4CgKbtvffei86dO8dLL70UTzzxRFx66aXxzjvvRLt27SIi4uc//3l8+9vfjpkzZ0bfvn3jyiuvjD/84Q8xffr00mO888470b1795gzZ07su+++5fpWACgjRz4A2MjcuXNj2LBhsffee0dNTU306tUrIiLefvvtmDNnThx88MGl8IiI+MIXvtDo/i+88ELMmDEj2rdvX/ro06dPRESjt28B0LK0LvcAADQ9Q4cOjZ49e8Ytt9wSdXV10dDQEAceeGCsXbt2q+6/YsWKGDp0aFx99dUb3bbHHnt81uMCsIMQHwA0snjx4pgzZ07ccsstMXDgwIiIeOKJJ0q39+7dO6ZMmRJr1qyJqqqqiIh45plnGj3GYYcdFv/7v/8bvXr1itat/VUDwN942xUAjeyyyy6x6667xoQJE+K1116LRx99NC666KLS7WeffXY0NDTEueeeG7Nnz47p06fHtddeGxERFRUVERExevToeP/992PYsGHxzDPPxOuvvx7Tp0+PkSNHxvr168vyfQFQfuIDgEYqKyvjv//7v+O5556LAw88MC688MK45pprSrfX1NTEfffdF7NmzYq+ffvGuHHj4rLLLouIKJ0HUldXF08++WSsX78+Bg0aFAcddFCMHTs2OnbsGJWV/uoBaKlc7QqAT+0Xv/hFjBw5MpYuXRrV1dXlHgeAJsobcQH4xO64447Ye++9Y88994wXXnghvvvd78aZZ54pPADYLPEBwCe2aNGiuOyyy2LRokWxxx57xNe+9rX40Y9+VO6xAGjivO0KAABI4aw/AAAghfgAAABSiA8AACCF+AAAAFKIDwAAIIX4AAAAUogPAAAghfgAAABS/D8wupwtZnO3UgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAANbBJREFUeJzt3Xl4VPW9+PFPEpIQWUWBksuuooJsCqikLlxRL4vV1q3uWLVatF5Ky60Wq6iota4VxUJ/FbQu12or3taFxY0r7griVnDBFQXrAqhsJuf3hzdTR7aA+A3B1+t58mjOnDnzOZMjzpuZc1KQZVkWAAAA37DC2h4AAAD4dhAfAABAEuIDAABIQnwAAABJiA8AACAJ8QEAACQhPgAAgCTEBwAAkIT4AAAAkhAfwGbv9ddfj4KCgpg4cWJtj5Ln3nvvjR49ekT9+vWjoKAgPv7449oeabX23nvv2Hvvvb+RbRcUFMRpp532jWybmmnfvn0MGTKktscAviXEB1BjEydOjIKCgryvFi1aRL9+/eKee+5JPs+DDz6YN0txcXF07Ngxjj322Hjttdc2ymM88sgjMWrUqI0eBh988EEcdthhUVZWFtdcc0386U9/igYNGmzw9m6++ea48sorN96AAPANqFfbAwB1z3nnnRcdOnSILMtiwYIFMXHixBg4cGD87W9/i8GDByef5/TTT4/evXvHypUr45lnnonx48fHXXfdFc8991yUl5d/rW0/8sgjce6558aQIUOiadOmG2fgiHjyySdjyZIlcf7550f//v2/9vZuvvnmeP7552PYsGFffzgA+IaID2C9DRgwIHr16pX7/oQTToiWLVvGLbfcUivxsccee8QhhxwSERHHH398dOrUKU4//fS4/vrr48wzz0w+T00sXLgwImKjBg1fT5ZlsWzZsigrK6vtUWrs888/j6qqqigpKantUQBqxMeugK+tadOmUVZWFvXq5f99xqeffho///nPo02bNlFaWhrbb799XHrppZFlWURELF26NHbYYYfYYYcdYunSpbn7ffjhh9GqVavo27dvVFZWrvc8//7v/x4REfPmzVvrevfff3/sscce0aBBg2jatGkceOCB8dJLL+VuHzVqVIwYMSIiIjp06JD7eNfrr7++1u3edtttscsuu0RZWVlsvfXWcfTRR8c777yTu33vvfeO4447LiIievfuHQUFBWv9zP2SJUti2LBh0b59+ygtLY0WLVrEvvvuG88880xue3fddVe88cYbuRnbt28fEf/6qNxXZ67+yNqDDz6Yt3z8+PGxzTbbRFlZWfTp0yf+93//N+/2Tz75JBo0aBD/+Z//ucqcb7/9dhQVFcVFF1201udndSZNmhQ77bRTlJaWRpcuXeLee+9dZZ2ZM2fGgAEDonHjxtGwYcPYZ5994rHHHstbZ9SoUVFQULDKfVf3PLRv3z4GDx4ckydPjl69ekVZWVmMGzcuIiKmTp0a3/3ud6Np06bRsGHD2H777eNXv/rVeu1T9fanTJmSO7enc+fO8de//nWVdT/++OMYNmxY7r+VbbfdNi6++OKoqqrKrVN97tKll14aV155ZWyzzTZRWloaL774Yo3mybIsRo8eHa1bt44tttgi+vXrFy+88MIq63344Yfxi1/8Irp27RoNGzaMxo0bx4ABA+LZZ5/NrfNNHQfA5s87H8B6W7RoUfzzn/+MLMti4cKFMWbMmPjkk0/i6KOPzq2TZVl873vfiwceeCBOOOGE6NGjR0yePDlGjBgR77zzTlxxxRVRVlYW119/fVRUVMTIkSPj8ssvj4iIU089NRYtWhQTJ06MoqKi9Z7v1VdfjYiIrbbaao3rTJs2LQYMGBAdO3aMUaNGxdKlS2PMmDFRUVERzzzzTLRv3z5+8IMfxNy5c+OWW26JK664IrbeeuuIiGjevPkatztx4sQ4/vjjo3fv3nHRRRfFggUL4ne/+13MmDEjZs6cGU2bNo2RI0fG9ttvH+PHj899hG2bbbZZ4zZPOeWUuP322+O0006Lzp07xwcffBAPP/xwvPTSS7HzzjvHyJEjY9GiRfH222/HFVdcERERDRs2XO/n7Y9//GOcfPLJ0bdv3xg2bFi89tpr8b3vfS+aNWsWbdq0yW33+9//ftx6661x+eWX5/18brnllsiyLI466qj1etyHH344/vrXv8bQoUOjUaNGcdVVV8XBBx8cb775Zu5n+MILL8Qee+wRjRs3jv/6r/+K4uLiGDduXOy9997x0EMPxa677rre+xsRMWfOnDjiiCPi5JNPjpNOOim23377eOGFF2Lw4MHRrVu3OO+886K0tDReeeWVmDFjxnpv/+WXX47DDz88TjnllDjuuONiwoQJceihh8a9994b++67b0REfPbZZ7HXXnvFO++8EyeffHK0bds2HnnkkTjzzDPj3XffXeVcngkTJsSyZcvixz/+cZSWlkazZs1qNMvZZ58do0ePjoEDB8bAgQPjmWeeif322y9WrFiRt95rr70WkyZNikMPPTQ6dOgQCxYsiHHjxsVee+0VL774YpSXl38jxwHwLZEB1NCECROyiFjlq7S0NJs4cWLeupMmTcoiIhs9enTe8kMOOSQrKCjIXnnlldyyM888MyssLMymT5+e3XbbbVlEZFdeeeU653nggQeyiMiuu+667P3338/mz5+f3XXXXVn79u2zgoKC7Mknn8yyLMvmzZuXRUQ2YcKE3H179OiRtWjRIvvggw9yy5599tmssLAwO/bYY3PLLrnkkiwisnnz5q1znhUrVmQtWrTIdtppp2zp0qW55X//+9+ziMjOPvvs3LLq57J6xrVp0qRJduqpp651nUGDBmXt2rVbZXn143x1/urn7oEHHsibvUePHtny5ctz640fPz6LiGyvvfbKLZs8eXIWEdk999yTt81u3brlrVcTEZGVlJTkHQ/PPvtsFhHZmDFjcssOOuigrKSkJHv11Vdzy+bPn581atQo23PPPXPLzjnnnGx1/2tb3fPQrl27LCKye++9N2/dK664IouI7P3331+vffmq6u3/5S9/yS1btGhR1qpVq6xnz565Zeeff37WoEGDbO7cuXn3P+OMM7KioqLszTffzLLsX8dx48aNs4ULF67XLAsXLsxKSkqyQYMGZVVVVbnlv/rVr7KIyI477rjcsmXLlmWVlZV59583b15WWlqanXfeebllG/M4AL49fOwKWG/XXHNNTJ06NaZOnRo33nhj9OvXL0488cS8j5PcfffdUVRUFKeffnrefX/+859HlmV5V8caNWpUdOnSJY477rgYOnRo7LXXXqvcb21+9KMfRfPmzaO8vDwGDRoUn376aVx//fV556V82bvvvhuzZs2KIUOG5P2tcbdu3WLfffeNu+++u8aP/WVPPfVULFy4MIYOHRr169fPLR80aFDssMMOcdddd23Qdps2bRqPP/54zJ8/f4PuXxPVs59yyil55w8MGTIkmjRpkrdu//79o7y8PG666abcsueffz5mz56d9+5XTfXv3z/vnZ9u3bpF48aNc1csq6ysjClTpsRBBx0UHTt2zK3XqlWrOPLII+Phhx+OxYsXr/fjRnzxcbr9998/b1n1eTh33nln3seeNkR5eXl8//vfz33fuHHjOPbYY2PmzJnx3nvvRcQXH9PbY489Ysstt4x//vOfua/+/ftHZWVlTJ8+PW+bBx988FrffVudadOmxYoVK+KnP/1p3sfSVneBgtLS0igs/OLlQWVlZXzwwQe5j55Vf9QvYuMfB8C3g/gA1lufPn2if//+0b9//zjqqKPirrvuis6dO8dpp52W+wjHG2+8EeXl5dGoUaO8++64446526uVlJTEddddF/PmzYslS5bEhAkTVvu5/TU5++yzY+rUqXH//ffH7NmzY/78+XHMMcescf3qx95+++1XuW3HHXeMf/7zn/Hpp5/W+PFrst0ddtghb5/Xx29/+9t4/vnno02bNtGnT58YNWrURruUcLXq2bbbbru85dWXL/6ywsLCOOqoo2LSpEnx2WefRUTETTfdFPXr149DDz10vR+7bdu2qyzbcsst46OPPoqIiPfffz8+++yzNf68qqqq4q233lrvx434Ij6+6vDDD4+Kioo48cQTo2XLlvHDH/4w/vznP29QiGy77barHMudOnWKiMidf/Lyyy/HvffeG82bN8/7qr4KWvXFCdY287qs6efbvHnz2HLLLfOWVVVVxRVXXBHbbbddlJaWxtZbbx3NmzeP2bNnx6JFi3LrbezjAPh2EB/A11ZYWBj9+vWLd999N15++eUN2sbkyZMjImLZsmXrvY2uXbtG//79o1+/ftG1a9dVTnyv6w477LB47bXXYsyYMVFeXh6XXHJJdOnSpUa/W2VNEbchJ/J/2bHHHhuffPJJTJo0KbIsi5tvvjkGDx68yrskNbGm83qy/7swwfpY3/1d3ZWtysrKYvr06TFt2rQ45phjYvbs2XH44YfHvvvu+7Wft9WpqqqKfffdN/du4le/Dj744HXOvDFdeOGFMXz48Nhzzz3jxhtvjMmTJ8fUqVOjS5cuqwTYxjwOgG+Hzev/0ECt+fzzzyPii6vgRES0a9cupk2bFkuWLMl79+Mf//hH7vZqs2fPjvPOOy+OP/74mDVrVpx44onx3HPPfWMvYKofe86cOavc9o9//CO23nrr3C/8W593YL683eorblWbM2dO3j6vr1atWsXQoUNj6NChsXDhwth5553jggsuiAEDBqx1zuq/1f7qL0n86rsw1bO9/PLLebOvXLky5s2bF927d89bf6eddoqePXvGTTfdFK1bt44333wzxowZs8H7tzbNmzePLbbYYo0/r8LCwtwJ8V/e3y9fxnh933UqLCyMffbZJ/bZZ5+4/PLL48ILL4yRI0fGAw88sF6/l+WVV16JLMvyfj5z586NiMhdkWybbbaJTz75ZKP8vpc1+fLP98vvZL3//vu5d5iq3X777dGvX7/44x//mLf8448/zl10oVrK4wDYPHjnA/jaVq5cGVOmTImSkpLcx6oGDhwYlZWVcfXVV+ete8UVV0RBQUHuRfPKlStjyJAhUV5eHr/73e9i4sSJsWDBgvjZz372jc3bqlWr6NGjR1x//fV5L8qff/75mDJlSgwcODC3rDpCavIbznv16hUtWrSI3//+97F8+fLc8nvuuSdeeumlGDRo0HrPWllZmfdRl4iIFi1aRHl5ed5jNGjQYJX1IiJ3LsWXzxuorKyM8ePHrzJ78+bN4/e//33e1Y8mTpy4xn0/5phjYsqUKXHllVfGVlttlfuZbmxFRUWx3377xZ133pl3qdwFCxbEzTffHN/97nejcePGEbH6/a0+B6imPvzww1WW9ejRIyIi7zmvifnz58cdd9yR+37x4sVxww03RI8ePeI73/lORHzxztajjz6ae/fvyz7++ONc2H8d/fv3j+Li4hgzZkzeO0pfvZJWxBfP91ffdbrtttvyLhf9ZamOA2Dz4J0PYL3dc889uXcwFi5cGDfffHO8/PLLccYZZ+ReBB5wwAHRr1+/GDlyZLz++uvRvXv3mDJlStx5550xbNiw3IvE0aNHx6xZs+K+++6LRo0aRbdu3eLss8+Os846Kw455JC8ENiYLrnkkhgwYEDsvvvuccIJJ+QutdukSZMYNWpUbr1ddtklIiJGjhwZP/zhD6O4uDgOOOCAXJR8WXFxcVx88cVx/PHHx1577RVHHHFE7lK77du336CgWrJkSbRu3ToOOeSQ6N69ezRs2DCmTZsWTz75ZFx22WV5c956660xfPjw6N27dzRs2DAOOOCA6NKlS+y2225x5plnxocffhjNmjWL//7v/17lBW1xcXGMHj06Tj755Pj3f//3OPzww2PevHkxYcKEVc75qHbkkUfGf/3Xf8Udd9wRP/nJT6K4uHi996+mRo8enfvdG0OHDo169erFuHHjYvny5fHb3/42t95+++0Xbdu2jRNOOCFGjBgRRUVFcd1110Xz5s3jzTffrNFjnXfeeTF9+vQYNGhQtGvXLhYuXBhjx46N1q1bx3e/+931mrtTp05xwgknxJNPPhktW7aM6667LhYsWBATJkzIrTNixIj4n//5nxg8eHAMGTIkdtlll/j000/jueeei9tvvz1ef/31Vd5xWF/NmzePX/ziF3HRRRfF4MGDY+DAgTFz5sy45557Vtn24MGDc+9E9u3bN5577rm46aabNonjANgM1OKVtoA6ZnWX2q1fv37Wo0eP7Nprr827hGeWZdmSJUuyn/3sZ1l5eXlWXFycbbfddtkll1ySW+/pp5/O6tWrl/30pz/Nu9/nn3+e9e7dOysvL88++uijNc5TfbnY2267ba1zr+5Su1mWZdOmTcsqKiqysrKyrHHjxtkBBxyQvfjii6vc//zzz8/+7d/+LSssLKzRZXdvvfXWrGfPnllpaWnWrFmz7KijjsrefvvtvHVqeqnd5cuXZyNGjMi6d++eNWrUKGvQoEHWvXv3bOzYsXnrffLJJ9mRRx6ZNW3aNIuIvMvuvvrqq1n//v2z0tLSrGXLltmvfvWrbOrUqXmX2q02duzYrEOHDllpaWnWq1evbPr06dlee+21xkunDhw4MIuI7JFHHlnrfqxJRKz2MsLt2rXLu/xrlmXZM888k+2///5Zw4YNsy222CLr16/fah/36aefznbdddespKQka9u2bXb55Zev8VK7gwYNWuX+9913X3bggQdm5eXlWUlJSVZeXp4dccQRq1wKd12qtz958uSsW7duWWlpabbDDjus9nhdsmRJduaZZ2bbbrttVlJSkm299dZZ3759s0svvTRbsWJFlmX/Oo4vueSS9ZqjWmVlZXbuuedmrVq1ysrKyrK99947e/7551d5rpctW5b9/Oc/z61XUVGRPfroo9/ocQB8exRk2Qac0QcAEfH9738/nnvuuXjllVdqe5RNTvv27WOnnXaKv//977U9yjfOcQDUlHM+ANgg7777btx1111rvawxmz/HAbA+nPMBwHqZN29ezJgxI/7f//t/UVxcHCeffPIq61T/Ar01KSsrq7OXY33//ffXesndkpKSvF9eubnOU5PjAOCrxAcA6+Whhx6K448/Ptq2bRvXX3997qpNX9aqVau1buO4446LiRMnfkMTfrN69+691kv37rXXXvHggw9u9vPU5DgA+Crn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U2PlypWx3377xaefflrbo8FqPfnkkzFu3Ljo1q1bbY8CeT766KOoqKiI4uLiuOeee+LFF1+Myy67LLbccsvaHg1yLr744rj22mvj6quvjpdeeikuvvji+O1vfxtjxoyp7dE2GXX+Uru77rpr9O7dO66++uqIiKiqqoo2bdrET3/60zjjjDNqeTrI9/7770eLFi3ioYceij333LO2x4E8n3zySey8884xduzYGD16dPTo0SOuvPLK2h4LIiLijDPOiBkzZsT//u//1vYosEaDBw+Oli1bxh//+MfcsoMPPjjKysrixhtvrMXJNh11+p2PFStWxNNPPx39+/fPLSssLIz+/fvHo48+WouTweotWrQoIiKaNWtWy5PAqk499dQYNGhQ3p+psKn4n//5n+jVq1cceuih0aJFi+jZs2f84Q9/qO2xIE/fvn3jvvvui7lz50ZExLPPPhsPP/xwDBgwoJYn23TUq+0Bvo5//vOfUVlZGS1btsxb3rJly/jHP/5RS1PB6lVVVcWwYcOioqIidtppp9oeB/L893//dzzzzDPx5JNP1vYosFqvvfZaXHvttTF8+PD41a9+FU8++WScfvrpUVJSEscdd1xtjwcR8cU7dIsXL44ddtghioqKorKyMi644II46qijanu0TUadjg+oS0499dR4/vnn4+GHH67tUSDPW2+9Ff/5n/8ZU6dOjfr169f2OLBaVVVV0atXr7jwwgsjIqJnz57x/PPPx+9//3vxwSbjz3/+c9x0001x8803R5cuXWLWrFkxbNiwKC8vd5z+nzodH1tvvXUUFRXFggUL8pYvWLAgvvOd79TSVLCq0047Lf7+97/H9OnTo3Xr1rU9DuR5+umnY+HChbHzzjvnllVWVsb06dPj6quvjuXLl0dRUVEtTggRrVq1is6dO+ct23HHHeMvf/lLLU0EqxoxYkScccYZ8cMf/jAiIrp27RpvvPFGXHTRReLj/9Tpcz5KSkpil112ifvuuy+3rKqqKu67777Yfffda3Ey+EKWZXHaaafFHXfcEffff3906NChtkeCVeyzzz7x3HPPxaxZs3JfvXr1iqOOOipmzZolPNgkVFRUrHKp8rlz50a7du1qaSJY1WeffRaFhfkvr4uKiqKqqqqWJtr01Ol3PiIihg8fHscdd1z06tUr+vTpE1deeWV8+umncfzxx9f2aBCnnnpq3HzzzXHnnXdGo0aN4r333ouIiCZNmkRZWVktTwdfaNSo0SrnITVo0CC22mor5yexyfjZz34Wffv2jQsvvDAOO+yweOKJJ2L8+PExfvz42h4Ncg444IC44IILom3bttGlS5eYOXNmXH755fGjH/2otkfbZNT5S+1GRFx99dVxySWXxHvvvRc9evSIq666KnbdddfaHguioKBgtcsnTJgQQ4YMSTsMrIe9997bpXbZ5Pz973+PM888M15++eXo0KFDDB8+PE466aTaHgtylixZEr/+9a/jjjvuiIULF0Z5eXkcccQRcfbZZ0dJSUltj7dJ2CziAwAA2PTV6XM+AACAukN8AAAASYgPAAAgCfEBAAAkIT4AAIAkxAcAAJCE+AAAAJIQHwAAQBLiA2ATMHHixGjatOlG2daDDz4YBQUF8fHHH2+U7X0beQ4BvhniA2ADDRkyJA466KDaHgMA6gzxAcBGkWVZfP7557U9xipWrlxZ2yMA8H/EB8A63H777dG1a9coKyuLrbbaKvr37x8jRoyI66+/Pu68884oKCiIgoKCePDBB1f7cZ1Zs2ZFQUFBvP7667llEydOjLZt28YWW2wR3//+9+ODDz7I3fb6669HYWFhPPXUU3lzXHnlldGuXbuoqqqq0dxPP/109OrVK7bYYovo27dvzJkzJ+/2a6+9NrbZZpsoKSmJ7bffPv70pz/lzVBQUBCzZs3KLfv4449z+xnxr48m3XPPPbHLLrtEaWlpPPzww/Hss89Gv379olGjRtG4cePYZZddVtmX1an+6NmkSZNiu+22i/r168f+++8fb731Vt56d955Z+y8885Rv3796NixY5x77rl50VNQUBDXXnttfO9734sGDRrEBRdcsM7Hvvvuu6NTp05RVlYW/fr1y/tZRUR88MEHccQRR8S//du/xRZbbBFdu3aNW265JXf7DTfcEFtttVUsX748734HHXRQHHPMMet8fIBvjQyANZo/f35Wr1697PLLL8/mzZuXzZ49O7vmmmuyJUuWZIcddlj2H//xH9m7776bvfvuu9ny5cuzBx54IIuI7KOPPsptY+bMmVlEZPPmzcuyLMsee+yxrLCwMLv44ouzOXPmZL/73e+ypk2bZk2aNMndZ999982GDh2aN0u3bt2ys88+e50zV8+w6667Zg8++GD2wgsvZHvssUfWt2/f3Dp//etfs+Li4uyaa67J5syZk1122WVZUVFRdv/992dZlmXz5s3LIiKbOXNm7j4fffRRFhHZAw88kPc43bp1y6ZMmZK98sor2QcffJB16dIlO/roo7OXXnopmzt3bvbnP/85mzVr1jrnnjBhQlZcXJz16tUre+SRR7Knnnoq69OnT97c06dPzxo3bpxNnDgxe/XVV7MpU6Zk7du3z0aNGpVbJyKyFi1aZNddd1326quvZm+88cZaH/fNN9/MSktLs+HDh2f/+Mc/shtvvDFr2bJl3s/x7bffzi655JJs5syZ2auvvppdddVVWVFRUfb4449nWZZln332WdakSZPsz3/+c267CxYsyOrVq5d7TgHIMvEBsBZPP/10FhHZ66+/vsptxx13XHbggQfmLatJfBxxxBHZwIED8+53+OGH58XHrbfemm255ZbZsmXLcnMUFBTktrE21TNMmzYtt+yuu+7KIiJbunRplmVZ1rdv3+ykk07Ku9+hhx6am2t94mPSpEl522nUqFE2ceLEdc75VRMmTMgiInvsscdyy1566aUsInIv8vfZZ5/swgsvzLvfn/70p6xVq1a57yMiGzZsWI0f98wzz8w6d+6ct+yXv/zlKj/Hrxo0aFD285//PPf9T37yk2zAgAG57y+77LKsY8eOWVVVVY1nAdjc+dgVwFp079499tlnn+jatWsceuih8Yc//CE++uijr7XNl156KXbddde8Zbvvvnve9wcddFAUFRXFHXfcERFffCSpX79+0b59+xo/Trdu3XL/3qpVq4iIWLhwYW6GioqKvPUrKiripZdeqvH2q/Xq1Svv++HDh8eJJ54Y/fv3j9/85jfx6quv1nhb9erVi969e+e+32GHHaJp06a5uZ599tk477zzomHDhrmvk046Kd5999347LPP1jjT2tTk51FZWRnnn39+dO3aNZo1axYNGzaMyZMnx5tvvplb56STToopU6bEO++8ExFf/MyGDBkSBQUFNZ4FYHMnPgDWoqioKKZOnRr33HNPdO7cOcaMGRPbb799zJs3b7XrFxZ+8cdqlmW5ZRtywnNJSUkce+yxMWHChFixYkXcfPPN8aMf/Wi9tlFcXJz79+oXwDU9X2R99qNBgwZ5348aNSpeeOGFGDRoUNx///3RuXPnXER9XZ988kmce+65MWvWrNzXc889Fy+//HLUr19/jTN9XZdcckn87ne/i1/+8pfxwAMPxKxZs2L//fePFStW5Nbp2bNndO/ePW644YZ4+umn44UXXoghQ4Zs1DkA6jrxAbAOBQUFUVFREeeee27MnDkzSkpK4o477oiSkpKorKzMW7d58+YREfHuu+/mln35pO2IiB133DEef/zxvGWPPfbYKo974oknxrRp02Ls2LHx+eefxw9+8IONtEdfzDBjxoy8ZTNmzIjOnTtHRM32Y206deoUP/vZz2LKlCnxgx/8ICZMmFCj+33++ed5J6fPmTMnPv7449hxxx0jImLnnXeOOXPmxLbbbrvKV3Uwra8dd9wxnnjiibxlX/15zJgxIw488MA4+uijo3v37tGxY8eYO3fuKts68cQTY+LEiTFhwoTo379/tGnTZoNmAthc1avtAQA2ZY8//njcd999sd9++0WLFi3i8ccfj/fffz923HHHWLZsWUyePDnmzJkTW221VTRp0iS23XbbaNOmTYwaNSouuOCCmDt3blx22WV52zz99NOjoqIiLr300jjwwANj8uTJce+9967y2DvuuGPstttu8ctf/jJ+9KMfRVlZ2UbbrxEjRsRhhx0WPXv2jP79+8ff/va3+Otf/xrTpk2LiIiysrLYbbfd4je/+U106NAhFi5cGGedddY6t7t06dIYMWJEHHLIIdGhQ4d4++2348knn4yDDz64RnMVFxfHT3/607jqqquiXr16cdppp8Vuu+0Wffr0iYiIs88+OwYPHhxt27aNQw45JAoLC+PZZ5+N559/PkaPHr1Bz8Upp5wSl112WYwYMSJOPPHEePrpp2PixIl562y33XZx++23xyOPPBJbbrllXH755bFgwYJcrFU78sgj4xe/+EX84Q9/iBtuuGGD5gHYrNX2SScAm7IXX3wx23///bPmzZtnpaWlWadOnbIxY8ZkWZZlCxcuzPbdd9+sYcOGeSdiP/zww1nXrl2z+vXrZ3vssUd222235Z1wnmVZ9sc//jFr3bp1VlZWlh1wwAHZpZdemnfC+ZfXi4jsiSeeqPHMNTnpPcuybOzYsVnHjh2z4uLirFOnTtkNN9ywyr7vvvvuWVlZWdajR49sypQpqz3h/MuPs3z58uyHP/xh1qZNm6ykpCQrLy/PTjvttNyJ7mszYcKErEmTJtlf/vKXrGPHjllpaWnWv3//Va5Wde+992Z9+/bNysrKssaNG2d9+vTJxo8fn7s9IrI77rijxs9XlmXZ3/72t2zbbbfNSktLsz322CO77rrr8vbtgw8+yA488MCsYcOGWYsWLbKzzjorO/bYY1e54ECWZdkxxxyTNWvWLHexAAD+pSDLvvSBXgA2Keeff37cdtttMXv27Noe5Rs3ceLEGDZsWN7vSKmL9tlnn+jSpUtcddVVtT0KwCbHx64ANkGffPJJvP7663H11Vdv8MeJSOujjz7K/aLJsWPH1vY4AJskJ5wDbIJOO+202GWXXWLvvfde5SpXp5xySt6lZr/8dcopp9TSxOs2YMCANc594YUXfmOPm+r56tmzZwwZMiQuvvji2H777TfadgE2Jz52BVDHLFy4MBYvXrza2xo3bhwtWrRIPFHNvPPOO7F06dLV3tasWbNo1qzZN/K4dfX5AtgciQ8AACAJH7sCAACSEB8AAEAS4gMAAEhCfAAAAEmIDwAAIAnxAQAAJCE+AACAJP4/lmBPzuYrLI8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAN2ZJREFUeJzt3XmUFeWd+OFvdwNNgyyy0wmrKCC4EDQGjaKGjCuawWg0JIJiNEIkOolJHKJgNEOMuwYXnBEc1MR1dFwQd9HITxHXuAAqSwKICrKoQAP9/v7wcMcbQBvEt0Wf5xyO3qq6t966Vbb3Q92qLkkppQAAAPicldb2AAAAgK8G8QEAAGQhPgAAgCzEBwAAkIX4AAAAshAfAABAFuIDAADIQnwAAABZiA8AACAL8QFstWbPnh0lJSUxfvz42h5Kkfvuuy923XXXqF+/fpSUlMSSJUtqe0ifq8GDB0fHjh1rexgAbAXEBxDjx4+PkpKSoj+tWrWK/fbbLyZOnJh9PI8++mjRWOrWrRudO3eOY489Nt58880tso4nn3wyRo0atcXDYNGiRXHUUUdFRUVFjBkzJiZMmBANGzb8TK/5yiuvxKhRo2L27Nnrzbviiiu+cPHF1uHee++NUaNG1fYwgK+YOrU9AOCL43e/+1106tQpUkqxcOHCGD9+fBx88MFx1113xaGHHpp9PMOHD4/dd989Vq9eHc8++2yMHTs27rnnnnjppZeisrLyM732k08+GWeffXYMHjw4mjZtumUGHBFTp06N5cuXxznnnBP9+vXbIq/5yiuvxNlnnx377rvvemcYrrjiimjRokUMHjx4i6yLr4577703xowZI0CArMQHUHDQQQfFbrvtVng8ZMiQaN26dfz5z3+ulfjYe++94/vf/35ERBx33HGxww47xPDhw+O6666LM844I/t4auLtt9+OiNiiQcOXU3V1dVRVVUX9+vVreygA2fjaFbBRTZs2jYqKiqhTp/jvKT744IP4xS9+Ee3atYvy8vLo2rVrXHDBBZFSioiIFStWRLdu3aJbt26xYsWKwvMWL14cbdu2jT333DPWrl27yePZf//9IyJi1qxZn7jcww8/HHvvvXc0bNgwmjZtGocffni8+uqrhfmjRo2K008/PSIiOnXqVPh614a+1vRxt9xyS/Tu3TsqKiqiRYsW8aMf/SjmzZtXmL/vvvvGoEGDIiJi9913j5KSkk88IzFnzpwYOnRodO3aNSoqKqJ58+Zx5JFHFo1j/PjxceSRR0ZExH777VcY66OPPhodO3aMl19+OR577LHC9H333bfw3CVLlsSpp55a2E9dunSJ8847L6qrqwvLrLtu5oILLoixY8fGdtttF+Xl5bH77rvH1KlT1xvzHXfcET179oz69etHz54943/+5382uG0XXHBB7LnnntG8efOoqKiI3r17x6233rreciUlJfGzn/2s8Lrl5eXRo0ePuO+++9Zbdt68eTFkyJCorKyM8vLy6NSpU5x88slRVVW1SdtcE/vuu2/07Nkzpk2bFnvuuWdUVFREp06d4qqrrlpv2VWrVsXIkSOjS5cuUV5eHu3atYtf/epXsWrVqg1u6w033BA9evSI8vLywnZuqW2r6f4cPHhwjBkzpjCudX/Wqen+W7FiRQwfPjxatGgRjRo1isMOOyzmzZsXJSUl651RmTdvXhx//PHRunXrwn6+9tprN2GvAF8GznwABUuXLo133303Ukrx9ttvx+WXXx7vv/9+/OhHPyosk1KKww47LB555JEYMmRI7LrrrjFp0qQ4/fTTY968eXHxxRdHRUVFXHfddbHXXnvFiBEj4qKLLoqIiGHDhsXSpUtj/PjxUVZWtsnje+ONNyIionnz5htd5sEHH4yDDjooOnfuHKNGjYoVK1bE5ZdfHnvttVc8++yz0bFjxxgwYEDMmDEj/vznP8fFF18cLVq0iIiIli1bbvR1x48fH8cdd1zsvvvuMXr06Fi4cGFceuml8de//jWee+65aNq0aYwYMSK6du0aY8eOLXyFbbvtttvoa06dOjWefPLJOProo+PrX/96zJ49O6688srYd99945VXXokGDRrEPvvsE8OHD4/LLrss/v3f/z26d+8eERHdu3ePSy65JE455ZTYZpttYsSIERER0bp164iI+PDDD6Nv374xb968OOmkk6J9+/bx5JNPxhlnnBELFiyISy65pGgsN954YyxfvjxOOumkKCkpiT/+8Y8xYMCAePPNN6Nu3boREXH//ffHEUccETvuuGOMHj06Fi1aFMcdd1x8/etfX2/bLr300jjssMNi4MCBUVVVFX/5y1/iyCOPjLvvvjsOOeSQomWfeOKJuP3222Po0KHRqFGjuOyyy+KII46IuXPnFvb1/Pnz45vf/GYsWbIkTjzxxOjWrVvMmzcvbr311vjwww+jXr16m7zNn+a9996Lgw8+OI466qg45phj4uabb46TTz456tWrF8cff3xEfHT24rDDDosnnngiTjzxxOjevXu89NJLcfHFF8eMGTPijjvuKHrNhx9+OG6++eb42c9+Fi1atIiOHTt+Ltv2afvzpJNOivnz58cDDzwQEyZM2Oz9N3jw4Lj55pvjxz/+cXzrW9+Kxx57bL39GxGxcOHC+Na3vlUIsJYtW8bEiRNjyJAhsWzZsjj11FM3ad8AW7EEfOWNGzcuRcR6f8rLy9P48eOLlr3jjjtSRKRzzz23aPr3v//9VFJSkl5//fXCtDPOOCOVlpamyZMnp1tuuSVFRLrkkks+dTyPPPJIioh07bXXpnfeeSfNnz8/3XPPPaljx46ppKQkTZ06NaWU0qxZs1JEpHHjxhWeu+uuu6ZWrVqlRYsWFaa98MILqbS0NB177LGFaeeff36KiDRr1qxPHU9VVVVq1apV6tmzZ1qxYkVh+t13350iIp111lmFaevey3Vj/CQffvjhetOmTJmSIiL993//d2HauvfukUceWW/5Hj16pL59+643/ZxzzkkNGzZMM2bMKJr+m9/8JpWVlaW5c+emlP7vPWzevHlavHhxYbk777wzRUS66667CtN23XXX1LZt27RkyZLCtPvvvz9FROrQocMnbltVVVXq2bNn2n///YumR0SqV69e0XHzwgsvpIhIl19+eWHasccem0pLSzf4vlZXV2/SNtdE3759U0SkCy+8sDBt1apVheOrqqoqpZTShAkTUmlpaXr88ceLnn/VVVeliEh//etfi7a1tLQ0vfzyy0XLbslt25T9OWzYsLSxjwE12X/Tpk1LEZFOPfXUomUHDx6cIiKNHDmyMG3IkCGpbdu26d133y1a9uijj05NmjTZ4H8LwJeTr10BBWPGjIkHHnggHnjggbj++utjv/32ixNOOCFuv/32wjL33ntvlJWVxfDhw4ue+4tf/CJSSkV3xxo1alT06NEjBg0aFEOHDo2+ffuu97xPcvzxx0fLli2jsrIyDjnkkPjggw/iuuuuK7ou5eMWLFgQzz//fAwePDiaNWtWmL7zzjvHd7/73bj33ntrvO6Pe+aZZ+Ltt9+OoUOHFn0//5BDDolu3brFPffcs1mvW1FRUfj31atXx6JFi6JLly7RtGnTePbZZzfrNde55ZZbYu+9945tt9023n333cKffv36xdq1a2Py5MlFy//gBz+IbbfdtvB47733jogo3F1s3Xs7aNCgaNKkSWG57373u7Hjjjt+4ra99957sXTp0th77703uF39+vUrOkO08847R+PGjQvrrq6ujjvuuCP69++/wX2/7utCm7rNn6ZOnTpx0kknFR7Xq1cvTjrppHj77bdj2rRphXV27949unXrVrTOdV8RfOSRR4pes2/fvkXv1+e1bZ+2Pz9NTfbfuq+MDR06tOi5p5xyStHjlFLcdttt0b9//0gpFY3/gAMOiKVLl37m4x3YevjaFVDwzW9+s+gD0DHHHBO9evWKn/3sZ3HooYdGvXr1Ys6cOVFZWRmNGjUqeu66rwPNmTOnMK1evXpx7bXXxu677x7169ePcePGFX2v/NOcddZZsffee0dZWVm0aNEiunfvvt71Jx+3bt1du3Zdb1737t1j0qRJ8cEHH2zyrW8/6XW7desWTzzxxCa93jorVqyI0aNHx7hx42LevHmFa2YiPvoK3Gcxc+bMePHFFzf6VbJ1F8av0759+6LH6z64vvfeexHxf+/B9ttvv95rde3adb0Pj3fffXece+658fzzzxdd+7Ch/f/P6163/nXrfuedd2LZsmXRs2fPDW7LOpu6zZ+msrJyvWNlhx12iIiPrq341re+FTNnzoxXX321xuvs1KlT0ePPa9s+bX9+mprsvzlz5kRpael629SlS5eix++8804sWbIkxo4dG2PHjq3R+IEvL/EBbFRpaWnst99+cemll8bMmTOjR48em/wakyZNioiIlStXxsyZM9f7oPJJdtpppy12u9ovolNOOSXGjRsXp556avTp0yeaNGkSJSUlcfTRR2/yBdL/rLq6Or773e/Gr371qw3OX/chep2NXYPz8SCqqccffzwOO+yw2GeffeKKK66Itm3bRt26dWPcuHFx4403rrf8llr3pm7zllBdXR077bRT4bqmf9auXbuixx8/o7Cp68m1Pzd1/9Vk7BERP/rRjwo3ZPhnO++88ya/LrB1Eh/AJ1qzZk1ERLz//vsREdGhQ4d48MEHY/ny5UVnP1577bXC/HVefPHF+N3vfhfHHXdcPP/883HCCSfESy+9VPS1nS1p3bqnT5++3rzXXnstWrRoUfib7E05A/Px1133dZp1pk+fXrTNm+LWW2+NQYMGxYUXXliYtnLlyvV+8eEnjXVj87bbbrt4//33t1i8rdvGmTNnrjfvn9/v2267LerXrx+TJk2K8vLywvRx48Zt1rpbtmwZjRs3jr/97W+fuNyW3ub58+evd6ZsxowZERGF37ey3XbbxQsvvBDf+c53NumYWqe2ti1i48dOTfdfhw4dorq6OmbNmlV0Ruz1118vWq5ly5bRqFGjWLt27Zf6LxOAmnHNB7BRq1evjvvvvz/q1atX+FrVwQcfHGvXro0//elPRctefPHFUVJSEgcddFDhuYMHD47Kysq49NJLY/z48bFw4cI47bTTPrfxtm3bNnbddde47rrrij7A/+1vf4v7778/Dj744MK0dR8oa/Ibznfbbbdo1apVXHXVVUVfQZk4cWK8+uqrG7y7T02UlZWt9zfRl19++Xq3If6ksTZs2HCD04866qiYMmVK4czTxy1ZsqQQlTX18ff2418Je+CBB+KVV14pWrasrCxKSkqKtmP27Nnr3fmppkpLS+N73/te3HXXXfHMM8+sN3/de7ilt3nNmjVx9dVXFx5XVVXF1VdfHS1btozevXsX1jlv3ry45ppr1nv+ihUr4oMPPvhCblvExo+rmu6/Aw44ICI++kWXH3f55Zev93pHHHFE3HbbbRuMrHfeeWeTxw5svZz5AAomTpxYOIPx9ttvx4033hgzZ86M3/zmN9G4ceOIiOjfv3/st99+MWLEiJg9e3bssssucf/998edd94Zp556auHC4XXfF3/ooYeiUaNGsfPOO8dZZ50Vv/3tb+P73/9+UQhsSeeff34cdNBB0adPnxgyZEjhVrtNmjQp+r0D6z48jhgxIo4++uioW7du9O/ff4PXg9StWzfOO++8OO6446Jv375xzDHHFG6127Fjx80OqkMPPTQmTJgQTZo0iR133DGmTJkSDz744Hq3Et51112jrKwszjvvvFi6dGmUl5fH/vvvH61atYrevXvHlVdeGeeee2506dIlWrVqFfvvv3+cfvrp8b//+79x6KGHxuDBg6N3797xwQcfxEsvvRS33nprzJ49u3CL4ZoaPXp0HHLIIfHtb387jj/++Fi8eHFcfvnl0aNHj8KZsYiPLsS/6KKL4sADD4wf/vCH8fbbb8eYMWOiS5cu8eKLL27We/Uf//Efcf/990ffvn0Lt7RdsGBB3HLLLfHEE09E06ZNt/g2V1ZWxnnnnRezZ8+OHXbYIW666aZ4/vnnY+zYsYXbD//4xz+Om2++OX7605/GI488EnvttVesXbs2Xnvttbj55ptj0qRJG71BQm1uW8T//TcwfPjwOOCAA6KsrCyOPvroGu+/3r17xxFHHBGXXHJJLFq0qHCr3XVnhz5+ZuUPf/hDPPLII7HHHnvET37yk9hxxx1j8eLF8eyzz8aDDz4Yixcv3qSxA1ux2rrNFvDFsaFb7davXz/tuuuu6corryzc7nOd5cuXp9NOOy1VVlamunXrpu233z6df/75heWmTZuW6tSpk0455ZSi561ZsybtvvvuqbKyMr333nsbHc+6W+3ecsstnzjuDd1qN6WUHnzwwbTXXnulioqK1Lhx49S/f//0yiuvrPf8c845J33ta19LpaWlNbrt7k033ZR69eqVysvLU7NmzdLAgQPTP/7xj6JlNuVWu++991467rjjUosWLdI222yTDjjggPTaa6+lDh06pEGDBhUte80116TOnTunsrKyotvuvvXWW+mQQw5JjRo1ShFRdNvd5cuXpzPOOCN16dIl1atXL7Vo0SLtueee6YILLijcKnbde3j++eevN774p9ulppTSbbfdlrp3757Ky8vTjjvumG6//fY0aNCg9W61+1//9V9p++23T+Xl5albt25p3LhxaeTIkevd2jUi0rBhw9Zb94begzlz5qRjjz02tWzZMpWXl6fOnTunYcOGpVWrVm3SNtdE3759U48ePdIzzzyT+vTpk+rXr586dOiQ/vSnP623bFVVVTrvvPNSjx49Unl5edp2221T796909lnn52WLl36qdu6JbdtU/bnmjVr0imnnJJatmyZSkpKivZNTfffBx98kIYNG5aaNWuWttlmm/S9730vTZ8+PUVE+sMf/lC07MKFC9OwYcNSu3btUt26dVObNm3Sd77znTR27NhP2BPAl01JSptxNSEAfIntu+++8e67737qtRis7/nnn49evXrF9ddfHwMHDqzt4QBfMK75AAA2y4oVK9abdskll0RpaWnss88+tTAi4IvONR8AfGUsXrw4qqqqNjq/rKxso79Lg/X98Y9/jGnTpsV+++0XderUiYkTJ8bEiRPjxBNPXO82wwAR4gOAr5ABAwbEY489ttH5HTp0iNmzZ+cb0FZuzz33jAceeCDOOeeceP/996N9+/YxatSoGDFiRG0PDfiCcs0HAF8Z06ZN+8Tf8l1RURF77bVXxhEBfLWIDwAAIAsXnAMAAFls9jUf1dXVMX/+/GjUqFHRLxICAAC+WlJKsXz58qisrIzS0o2f39js+Jg/f747WQAAAAV///vf4+tf//pG5292fDRq1KiwgsaNG2/uywAAAFu5ZcuWRbt27QqNsDGbHR/rvmrVuHFj8QEAAHzq5RguOAcAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBZ1ansAAGzdFi5cGEuXLq3tYcBWq0mTJtG6devaHgZkIT4A2GwLFy6MH/342Fhdtaq2hwJbrbr1yuP6Cf8tQPhKEB8AbLalS5fG6qpVsaJz36iu36S2h8PHlK5YEhWzJseKTvtEdUXT2h4OG1G6cmnEm4/F0qVLxQdfCeIDgM+sun6TqG7YoraHwQZUVzS1b4AvDBecAwAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8bAVWrlwZM2bMiJUrV9b2UAAA+ILYGj8jio+twNy5c+PEE0+MuXPn1vZQAAD4gtgaPyOKDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMiiTm0P4LNau3ZtvPjii7F48eJo1qxZ7LzzzlFWVvaZl92SzwUAALby+Jg8eXJcccUV8dZbbxWmtWnTJoYOHRr77LPPZi/7WdYDAABs2Fb7tavJkyfHyJEjo3PnzjFmzJi49957Y8yYMdG5c+cYOXJkTJ48ebOW/SzrAQAANm6rjI+1a9fGFVdcEX369Ilzzz03evToEQ0aNIgePXrEueeeG3369Ikrr7wy1q5du0nLfpb1AAAAn6zGX7tatWpVrFq1qvB42bJln8uAauLFF1+Mt956K84888woLS3up9LS0hg4cGAMGzYsXnzxxYiIGi/bq1evzV7PPz/38zBnzpzPfR0Am8LPJdgy/LfE5tgaj5sax8fo0aPj7LPP/jzHUmOLFy+OiIhOnTptcP666euW29RlP8t6Pk+///3vs6wHAMjL/+P5qqhxfJxxxhnxb//2b4XHy5Yti3bt2n0ug/o0zZo1i4iIWbNmRY8ePdabP2vWrKLlNnXZz7Kez9OIESOiQ4cOWdYFUBNz5szxoQm2AP+PZ3NsjT+Daxwf5eXlUV5e/nmOpcZ23nnnaNOmTdxwww1x7rnnFn0lqrq6Om644YZo27Zt7LzzzhERm7TsZ1nP561Dhw6xww47ZFkXAJCP/8fzVbFVXnBeVlYWQ4cOjSlTpsRvf/vbePnll+PDDz+Ml19+OX7729/GlClT4uSTT46ysrJNWvazrAcAAPhkW+3v+dhnn33i7LPPjiuuuCKGDRtWmN62bds4++yzi37/xqYs+1nWAwAAbNxWGx8RH4XBXnvtVaPfPL4py27J5wIAAB/ZquMj4qOvRtX0NrebsuyWfC4AALCVXvMBAABsfcQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmIj61A+/btY+zYsdG+ffvaHgoAAF8QW+NnxDq1PQA+Xf369WOHHXao7WEAAPAFsjV+RnTmAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBZ1ansAAGz9Slcure0h8E9KVywp+idfTP7b4atGfACw2Zo0aRJ165VHvPlYbQ+FjaiYNbm2h8CnqFuvPJo0aVLbw4AsxAcAm61169Zx/YT/jqVL/e0tbK4mTZpE69ata3sYkIX4AOAzad26tQ9OANSIC84BAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABkIT4AAIAsxAcAAJCF+AAAALKos7lPTClFRMSyZcu22GAAAICtz7omWNcIG7PZ8bF8+fKIiGjXrt3mvgQAAPAlsnz58mjSpMlG55ekT8uTjaiuro758+dHo0aNoqSkZLMHyGe3bNmyaNeuXfz973+Pxo0b1/Zw+Ipx/FGbHH/UNscgtemLdPyllGL58uVRWVkZpaUbv7Jjs898lJaWxte//vXNfTqfg8aNG9f6gcdXl+OP2uT4o7Y5BqlNX5Tj75POeKzjgnMAACAL8QEAAGQhPr4EysvLY+TIkVFeXl7bQ+EryPFHbXL8Udscg9SmrfH42+wLzgEAADaFMx8AAEAW4gMAAMhCfAAAAFmIDwAAIAvxsRWZN29e/OhHP4rmzZtHRUVF7LTTTvHMM88U5qeU4qyzzoq2bdtGRUVF9OvXL2bOnFmLI+bLpGPHjlFSUrLen2HDhkVExMqVK2PYsGHRvHnz2GabbeKII46IhQsX1vKo+bJYu3ZtnHnmmdGpU6eoqKiI7bbbLs4555z4+D1T/Azk87R8+fI49dRTo0OHDlFRURF77rlnTJ06tTDf8ceWMnny5Ojfv39UVlZGSUlJ3HHHHUXza3KsLV68OAYOHBiNGzeOpk2bxpAhQ+L999/PuBUbJz62Eu+9917stddeUbdu3Zg4cWK88sorceGFF8a2225bWOaPf/xjXHbZZXHVVVfFU089FQ0bNowDDjggVq5cWYsj58ti6tSpsWDBgsKfBx54ICIijjzyyIiIOO200+Kuu+6KW265JR577LGYP39+DBgwoDaHzJfIeeedF1deeWX86U9/ildffTXOO++8+OMf/xiXX355YRk/A/k8nXDCCfHAAw/EhAkT4qWXXop/+Zd/iX79+sW8efMiwvHHlvPBBx/ELrvsEmPGjNng/JocawMHDoyXX345Hnjggbj77rtj8uTJceKJJ+bahE+W2Cr8+te/Tt/+9rc3Or+6ujq1adMmnX/++YVpS5YsSeXl5enPf/5zjiHyFfPzn/88bbfddqm6ujotWbIk1a1bN91yyy2F+a+++mqKiDRlypRaHCVfFoccckg6/vjji6YNGDAgDRw4MKXkZyCfrw8//DCVlZWlu+++u2j6N77xjTRixAjHH5+biEj/8z//U3hck2PtlVdeSRGRpk6dWlhm4sSJqaSkJM2bNy/b2DfGmY+txP/+7//GbrvtFkceeWS0atUqevXqFddcc01h/qxZs+Ktt96Kfv36FaY1adIk9thjj5gyZUptDJkvsaqqqrj++uvj+OOPj5KSkpg2bVqsXr266Pjr1q1btG/f3vHHFrHnnnvGQw89FDNmzIiIiBdeeCGeeOKJOOiggyLCz0A+X2vWrIm1a9dG/fr1i6ZXVFTEE0884fgjm5oca1OmTImmTZvGbrvtVlimX79+UVpaGk899VT2Mf8z8bGVePPNN+PKK6+M7bffPiZNmhQnn3xyDB8+PK677rqIiHjrrbciIqJ169ZFz2vdunVhHmwpd9xxRyxZsiQGDx4cER8df/Xq1YumTZsWLef4Y0v5zW9+E0cffXR069Yt6tatG7169YpTTz01Bg4cGBF+BvL5atSoUfTp0yfOOeecmD9/fqxduzauv/76mDJlSixYsMDxRzY1OdbeeuutaNWqVdH8OnXqRLNmzb4Qx2Od2h4ANVNdXR277bZb/Md//EdERPTq1Sv+9re/xVVXXRWDBg2q5dHxVfNf//VfcdBBB0VlZWVtD4WviJtvvjluuOGGuPHGG6NHjx7x/PPPx6mnnhqVlZV+BpLFhAkT4vjjj4+vfe1rUVZWFt/4xjfimGOOiWnTptX20GCr4szHVqJt27ax4447Fk3r3r17zJ07NyIi2rRpExGx3t2FFi5cWJgHW8KcOXPiwQcfjBNOOKEwrU2bNlFVVRVLliwpWtbxx5Zy+umnF85+7LTTTvHjH/84TjvttBg9enRE+BnI52+77baLxx57LN5///34+9//Hk8//XSsXr06Onfu7Pgjm5oca23atIm33367aP6aNWti8eLFX4jjUXxsJfbaa6+YPn160bQZM2ZEhw4dIiKiU6dO0aZNm3jooYcK85ctWxZPPfVU9OnTJ+tY+XIbN25ctGrVKg455JDCtN69e0fdunWLjr/p06fH3LlzHX9sER9++GGUlhb/L6usrCyqq6sjws9A8mnYsGG0bds23nvvvZg0aVIcfvjhjj+yqcmx1qdPn1iyZEnRWbmHH344qqurY4899sg+5vXU9hXv1MzTTz+d6tSpk37/+9+nmTNnphtuuCE1aNAgXX/99YVl/vCHP6SmTZumO++8M7344ovp8MMPT506dUorVqyoxZHzZbJ27drUvn379Otf/3q9eT/96U9T+/bt08MPP5yeeeaZ1KdPn9SnT59aGCVfRoMGDUpf+9rX0t13351mzZqVbr/99tSiRYv0q1/9qrCMn4F8nu677740ceLE9Oabb6b7778/7bLLLmmPPfZIVVVVKSXHH1vO8uXL03PPPZeee+65FBHpoosuSs8991yaM2dOSqlmx9qBBx6YevXqlZ566qn0xBNPpO233z4dc8wxtbVJRcTHVuSuu+5KPXv2TOXl5albt25p7NixRfOrq6vTmWeemVq3bp3Ky8vTd77znTR9+vRaGi1fRpMmTUoRscHjasWKFWno0KFp2223TQ0aNEj/+q//mhYsWFALo+TLaNmyZennP/95at++fapfv37q3LlzGjFiRFq1alVhGT8D+TzddNNNqXPnzqlevXqpTZs2adiwYWnJkiWF+Y4/tpRHHnkkRcR6fwYNGpRSqtmxtmjRonTMMcekbbbZJjVu3Dgdd9xxafny5bWwNesrSeljvx4WAADgc+KaDwAAIAvxAQAAZCE+AACALMQHAACQhfgAAACyEB8AAEAW4gMAAMhCfAAAAFmID4AvuI4dO8Yll1xS28MAgM9MfABsxOzZs6OkpCSef/75oumDBw+O733ve7UyJj5/++67b5x66qm1PQyALyXxAcBWYfXq1bU9BAA+I/EBfKXdd9998e1vfzuaNm0azZs3j0MPPTTeeOONiIjo1KlTRET06tUrSkpKYt99941Ro0bFddddF3feeWeUlJRESUlJPProoxER8fe//z2OOuqoaNq0aTRr1iwOP/zwmD17dmFd686YXHDBBdG2bdto3rx5DBs2rOhD9dtvvx39+/ePioqK6NSpU9xwww3rjfmiiy6KnXbaKRo2bBjt2rWLoUOHxvvvv1+YP378+GjatGlMmjQpunfvHttss00ceOCBsWDBgqLXufbaa6NHjx5RXl4ebdu2jZ/97GeFeUuWLIkTTjghWrZsGY0bN479998/XnjhhRq9p6NGjYpdd901rr766mjXrl00aNAgjjrqqFi6dGnRcv/5n/8Z3bt3j/r160e3bt3iiiuuKMxbd9bppptuir59+0b9+vUL78VnGfe6sU2YMCE6duwYTZo0iaOPPjqWL19e2EePPfZYXHrppYX9O3v27Fi7dm0MGTIkOnXqFBUVFdG1a9e49NJLi7ZnzZo1MXz48MKx9Otf/zoGDRpUdJasuro6Ro8eXXidXXbZJW699dYava8AXwoJ4Cvs1ltvTbfddluaOXNmeu6551L//v3TTjvtlNauXZuefvrpFBHpwQcfTAsWLEiLFi1Ky5cvT0cddVQ68MAD04IFC9KCBQvSqlWrUlVVVerevXs6/vjj04svvpheeeWV9MMf/jB17do1rVq1KqWU0qBBg1Ljxo3TT3/60/Tqq6+mu+66KzVo0CCNHTu2MJ6DDjoo7bLLLmnKlCnpmWeeSXvuuWeqqKhIF198cWGZiy++OD388MNp1qxZ6aGHHkpdu3ZNJ598cmH+uHHjUt26dVO/fv3S1KlT07Rp01L37t3TD3/4w8IyV1xxRapfv3665JJL0vTp09PTTz9dtI5+/fql/v37p6lTp6YZM2akX/ziF6l58+Zp0aJFn/qejhw5MjVs2DDtv//+6bnnnkuPPfZY6tKlS9H6r7/++tS2bdt02223pTfffDPddtttqVmzZmn8+PEppZRmzZqVIiJ17NixsMz8+fM/87hHjhyZttlmmzRgwID00ksvpcmTJ6c2bdqkf//3f08ppbRkyZLUp0+f9JOf/KSwf9esWZOqqqrSWWedlaZOnZrefPPNdP3116cGDRqkm266qbDuc889NzVr1izdfvvt6dVXX00//elPU+PGjdPhhx9etEy3bt3Sfffdl9544400bty4VF5enh599NFPfV8BvgzEB8DHvPPOOyki0ksvvVT4APzcc88VLTNo0KCiD5QppTRhwoTUtWvXVF1dXZi2atWqVFFRkSZNmlR4XocOHdKaNWsKyxx55JHpBz/4QUoppenTp6eISE8//XRh/quvvpoiougD9j+75ZZbUvPmzQuPx40blyIivf7664VpY8aMSa1bty48rqysTCNGjNjg6z3++OOpcePGaeXKlUXTt9tuu3T11VdvdBzrjBw5MpWVlaV//OMfhWkTJ05MpaWlacGCBYXXuvHGG4ued84556Q+ffqklP4vPi655JKiZT7ruEeOHJkaNGiQli1bVph/+umnpz322KPwuG/fvunnP//5p27nsGHD0hFHHFF43Lp163T++ecXHq9Zsya1b9++cKysXLkyNWjQID355JNFrzNkyJB0zDHHfOr6AL4M6tTeOReA2jdz5sw466yz4qmnnop33303qqurIyJi7ty5seOOO9b4dV544YV4/fXXo1GjRkXTV65cWfgaV0REjx49oqysrPC4bdu28dJLL0VExKuvvhp16tSJ3r17F+Z369YtmjZtWvSaDz74YIwePTpee+21WLZsWaxZsyZWrlwZH374YTRo0CAiIho0aBDbbbdd0XrefvvtiPjoq13z58+P73znOxvdlvfffz+aN29eNH3FihVF2/JJ2rdvH1/72tcKj/v06RPV1dUxffr0aNSoUbzxxhsxZMiQ+MlPflJYZs2aNdGkSZOi19ltt90K/76lxt2xY8ei/fTx9+aTjBkzJq699tqYO3durFixIqqqqmLXXXeNiIilS5fGwoUL45vf/GZh+bKysujdu3fhmHr99dfjww8/jO9+97tFr1tVVRW9evX61PUDfBmID+ArrX///tGhQ4e45pprorKyMqqrq6Nnz55RVVW1Sa/z/vvvR+/evTd4jUbLli0L/163bt2ieSUlJYUPpzUxe/bsOPTQQ+Pkk0+O3//+99GsWbN44oknYsiQIVFVVVWIjw2tJ6UUEREVFRWfui1t27YtXMvycf8cQptj3fUp11xzTeyxxx5F8z4eZhERDRs2LPz7lhr35uyDv/zlL/HLX/4yLrzwwujTp080atQozj///Hjqqac+8Xn/PL6IiHvuuacozCIiysvLa/w6AFsz8QF8ZS1atCimT58e11xzTey9994REfHEE08U5terVy8iItauXVv0vHr16q037Rvf+EbcdNNN0apVq2jcuPFmjadbt26xZs2amDZtWuy+++4RETF9+vRYsmRJYZlp06ZFdXV1XHjhhVFa+tE9Q26++eZNWk+jRo2iY8eO8dBDD8V+++233vxvfOMb8dZbb0WdOnWiY8eOm7Utc+fOjfnz50dlZWVERPy///f/orS0NLp27RqtW7eOysrKePPNN2PgwIFfqHFHbHj//vWvf40999wzhg4dWpj28bMpTZo0idatW8fUqVNjn332iYiPjptnn322cHZkxx13jPLy8pg7d2707dt3s8cHsDVztyvgK2vbbbeN5s2bx9ixY+P111+Phx9+OP7t3/6tML9Vq1ZRUVER9913XyxcuLBwt6aOHTvGiy++GNOnT4933303Vq9eHQMHDowWLVrE4YcfHo8//njMmjUrHn300Rg+fHj84x//qNF4unbtGgceeGCcdNJJ8dRTT8W0adPihBNOKPob/y5dusTq1avj8ssvjzfffDMmTJgQV1111SZv+6hRo+LCCy+Myy67LGbOnBnPPvtsXH755RER0a9fv+jTp09873vfi/vvvz9mz54dTz75ZIwYMSKeeeaZGr1+/fr1Y9CgQfHCCy/E448/HsOHD4+jjjoq2rRpExERZ599dowePTouu+yymDFjRrz00ksxbty4uOiii2p13BEf7d+nnnoqZs+eXfgq3vbbbx/PPPNMTJo0KWbMmBFnnnlmTJ06teh5p5xySowePTruvPPOmD59evz85z+P9957L0pKSiLio3j65S9/Gaeddlpcd9118cYbbxTGf91119V4fABbM/EBfGWVlpbGX/7yl5g2bVr07NkzTjvttDj//PML8+vUqROXXXZZXH311VFZWRmHH354RET85Cc/ia5du8Zuu+0WLVu2jL/+9a/RoEGDmDx5crRv3z4GDBgQ3bt3jyFDhsTKlSs36UzIuHHjorKyMvr27RsDBgyIE088MVq1alWYv8suu8RFF10U5513XvTs2TNuuOGGGD169CZv+6BBg+KSSy6JK664Inr06BGHHnpozJw5MyI++hrSvffeG/vss08cd9xxscMOO8TRRx8dc+bMidatW9fo9bt06RIDBgyIgw8+OP7lX/4ldt5556Jb6Z5wwgnxn//5nzFu3LjYaaedom/fvjF+/PjC7Y1ra9wREb/85S+jrKwsdtxxx2jZsmXMnTs3TjrppBgwYED84Ac/iD322CMWLVpUdBYkIuLXv/51HHPMMXHsscdGnz59YptttokDDjgg6tevX1jmnHPOiTPPPDNGjx4d3bt3jwMPPDDuueeeT91ugC+LkrTuS8AAsAWMGjUq7rjjjvV+M/xXTXV1dXTv3j2OOuqoOOecc2p7OABfCK75AIAtYM6cOXH//fdH3759Y9WqVfGnP/0pZs2aFT/84Q9re2gAXxjiA4BN0qNHj5gzZ84G51199dWZR/PFUVpaGuPHj49f/vKXkVKKnj17xoMPPhjdu3ev7aEBfGH42hUAm2TOnDmxevXqDc5r3br1er/rBADWER8AAEAW7nYFAABkIT4AAIAsxAcAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWfx/CS6xDtDXDr0AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAALh1JREFUeJzt3XmY1XXd+P/XMDCLMAy4MDCKw5IsKaKoGZI/xA0ByS4VlcuFAU1LCVHzm0YqiUpqKIYl0a2Cu+GdmiKCFGouGS4YlgsqibcoaLKLIDOf3x/dzM0IKCC+zwCPx3XNVedzPuec15kPyHnOZ5m8LMuyAAAA+JrVy/UAAADA9kF8AAAASYgPAAAgCfEBAAAkIT4AAIAkxAcAAJCE+AAAAJIQHwAAQBLiAwAASEJ8AOTQv/71r8jLy4vx48fnepRaHn300dhnn32iqKgo8vLyYtGiRZv9XJWVldGqVastNtvX7ZBDDom99tor12MAbJPEB7BNGD9+fOTl5dX6atasWfTo0SMmT56cfJ7HH3+81iwNGjSINm3axGmnnRZvv/32FnmNZ555JoYPH/6VwmB9/v3vf8cJJ5wQxcXF8etf/zpuv/32aNiw4RZ9DQC2T/VzPQDAlnT55ZdH69atI8uymD9/fowfPz569+4dDz30UBx99NHJ5xkyZEgccMAB8dlnn8WLL74Y48aNi0mTJsWsWbOivLz8Kz33M888Ez//+c+jsrIymjRpsmUGjogZM2bE0qVLY8SIEXH44YdvsecFAPEBbFN69eoV+++/f83t008/PcrKyuLuu+/OSXwcfPDBcfzxx0dExMCBA6Ndu3YxZMiQmDBhQlx88cXJ59kYCxYsiIjYokHD5lu+fLk9T8A2w2FXwDatSZMmUVxcHPXr1/5Zy/Lly+OCCy6Ili1bRmFhYbRv3z5++ctfRpZlERGxYsWK6NChQ3To0CFWrFhR87iPP/44WrRoEQcddFBUVVVt8jyHHnpoRETMmTPnC9f785//HAcffHA0bNgwmjRpEsccc0y8+uqrNfcPHz48LrzwwoiIaN26dc3hXf/617++8HknTpwY++23XxQXF8fOO+8cp5xySrz33ns19x9yyCExYMCAiIg44IADIi8vLyorKzf4fEuXLo2hQ4dGq1atorCwMJo1axZHHHFEvPjii184R3V1dYwePTr23HPPKCoqirKysjjrrLNi4cKF66w7efLkmu9FSUlJ9OnTJ/7xj3/UWqeysjIaNWoUb7/9dvTs2TMaNmwY5eXlcfnll9ds0031z3/+M3r06BE77LBD7LrrrnHNNdess86CBQtqAreoqCg6d+4cEyZMqLXOmkPwHn/88VrL13e+z5r38dZbb0Xv3r2jpKQkTj755IiImD17dhx33HHRvHnzKCoqit122y1OOumkWLx48Wa9P4BcsOcD2KYsXrw4Pvroo8iyLBYsWBBjxoyJZcuWxSmnnFKzTpZl8d3vfjemT58ep59+euyzzz4xZcqUuPDCC+O9996L66+/PoqLi2PChAnRrVu3GDZsWFx33XUREXHOOefE4sWLY/z48ZGfn7/J87311lsREbHTTjttcJ1p06ZFr169ok2bNjF8+PBYsWJFjBkzJrp16xYvvvhitGrVKo499th444034u67747rr78+dt5554iI2GWXXTb4vOPHj4+BAwfGAQccECNHjoz58+fHDTfcEE8//XS89NJL0aRJkxg2bFi0b98+xo0bV3MIW9u2bTf4nD/4wQ/ivvvui8GDB8c3v/nN+Pe//x1PPfVUvPrqq9GlS5cNPu6ss86qmWfIkCExZ86cuPHGG+Oll16Kp59+Oho0aBAREbfffnsMGDAgevbsGVdffXV88skncdNNN8V3vvOdeOmll2qdyF5VVRVHHXVUfPvb345rrrkmHn300bjsssti9erVcfnll29wlvVZuHBhHHXUUXHsscfGCSecEPfdd1/85Cc/iU6dOkWvXr0i4j+Besghh8Sbb74ZgwcPjtatW8fEiROjsrIyFi1aFOeee+4mveYaq1evjp49e8Z3vvOd+OUvfxk77LBDrFq1Knr27BkrV66MH/3oR9G8efN477334uGHH45FixZFaWnpZr0WQHIZwDbg1ltvzSJina/CwsJs/PjxtdZ94IEHsojIrrjiilrLjz/++CwvLy978803a5ZdfPHFWb169bInn3wymzhxYhYR2ejRo790nunTp2cRkd1yyy3Zhx9+mM2bNy+bNGlS1qpVqywvLy+bMWNGlmVZNmfOnCwisltvvbXmsfvss0/WrFmz7N///nfNspdffjmrV69edtppp9Usu/baa7OIyObMmfOl86xatSpr1qxZttdee2UrVqyoWf7www9nEZFdeumlNcvWfC/XzPhFSktLs3POOecL1xkwYEBWUVFRc/svf/lLFhHZnXfeWWu9Rx99tNbypUuXZk2aNMm+//3v11rvgw8+yEpLS2stHzBgQBYR2Y9+9KOaZdXV1VmfPn2ygoKC7MMPP/zS97JG9+7ds4jIbrvttpplK1euzJo3b54dd9xxNctGjx6dRUR2xx131CxbtWpV1rVr16xRo0bZkiVLsiz7vz8L06dPr/U669v2a97HRRddVGvdl156KYuIbOLEiRv9PgDqIoddAduUX//61/HYY4/FY489FnfccUf06NEjzjjjjPjDH/5Qs84jjzwS+fn5MWTIkFqPveCCCyLLslpXxxo+fHjsueeeMWDAgDj77LOje/fu6zzuiwwaNCh22WWXKC8vjz59+sTy5ctjwoQJtc5LWdv7778fM2fOjMrKythxxx1rlu+9995xxBFHxCOPPLLRr722559/PhYsWBBnn312FBUV1Szv06dPdOjQISZNmrRZz9ukSZN47rnnYt68eRv9mIkTJ0ZpaWkcccQR8dFHH9V87bffftGoUaOYPn16REQ89thjsWjRoujfv3+t9fLz8+PAAw+sWW9tgwcPrvn/eXl5MXjw4Fi1alVMmzZtk95Xo0aNau0tKygoiG9961u1rlT2yCOPRPPmzaN///41yxo0aBBDhgyJZcuWxRNPPLFJr7m2H/7wh7Vur9mzMWXKlPjkk082+3kBcs1hV8A25Vvf+latD/b9+/ePfffdNwYPHhxHH310FBQUxDvvvBPl5eVRUlJS67EdO3aMiIh33nmnZllBQUHccsstccABB0RRUVHceuutkZeXt9HzXHrppXHwwQdHfn5+7LzzztGxY8d1zj9Z25rXbt++/Tr3dezYMaZMmbJZJyB/0fN26NAhnnrqqU16vjWuueaaGDBgQLRs2TL222+/6N27d5x22mnRpk2bDT5m9uzZsXjx4mjWrNl6719zwvvs2bMj4v/Ok/m8xo0b17pdr169dV63Xbt2ERFfei7M5+22227rbOemTZvG3//+95rb77zzTuyxxx5Rr17tn+Ot78/Rpqhfv37stttutZa1bt06zj///LjuuuvizjvvjIMPPji++93vximnnOKQK2CrIj6AbVq9evWiR48eccMNN8Ts2bNjzz333OTnmDJlSkREfPrppzF79uxo3br1Rj+2U6dO2/Tlak844YQ4+OCD4/7774+pU6fGtddeG1dffXX84Q9/qDk34vOqq6ujWbNmceedd673/jXnrVRXV0fEf877aN68+TrrfVHEfVUbOp8n24yT1zcUqxu6YEFhYeE6QRMRMWrUqKisrIwHH3wwpk6dGkOGDImRI0fGX//613ViBaCuEh/ANm/16tUREbFs2bKIiKioqIhp06bF0qVLa+39eO2112ruX+Pvf/97XH755TFw4MCYOXNmnHHGGTFr1qyv7afNa1779ddfX+e+1157LXbeeeeavR6bsgdm7ef9/J6E119/vdZ73lQtWrSIs88+O84+++xYsGBBdOnSJa688soNxkfbtm1j2rRp0a1btyguLt7g86450b1Zs2YbFXDV1dXx9ttv1+ztiIh44403IiK+lt+wXlFREX//+9+jurq6Vix8/s9R06ZNIyLW+WWQm7NnpFOnTtGpU6f42c9+Fs8880x069Ytxo4dG1dcccVmvguAtJzzAWzTPvvss5g6dWoUFBTUHA7Tu3fvqKqqihtvvLHWutdff33k5eXVfGj+7LPPorKyMsrLy+OGG26I8ePHx/z58+O888772uZt0aJF7LPPPjFhwoRaH1ZfeeWVmDp1avTu3btm2ZoI2ZjfcL7//vtHs2bNYuzYsbFy5cqa5ZMnT45XX301+vTps8mzVlVVrXOZ12bNmkV5eXmt1/i8E044IaqqqmLEiBHr3Ld69eqa99OzZ89o3LhxXHXVVfHZZ5+ts+6HH364zrK1t2mWZXHjjTdGgwYN4rDDDtvYt7XRevfuHR988EHce++9teYfM2ZMNGrUKLp37x4R/4mQ/Pz8ePLJJ2s9/je/+c1Gv9aSJUtqInqNTp06Rb169b7wew1Q19jzAWxTJk+eXPOT5wULFsRdd90Vs2fPjosuuqjmHIG+fftGjx49YtiwYfGvf/0rOnfuHFOnTo0HH3wwhg4dWvMT9yuuuCJmzpwZf/rTn6KkpCT23nvvuPTSS+NnP/tZHH/88bVCYEu69tpro1evXtG1a9c4/fTTay61W1paGsOHD69Zb7/99ouIiGHDhsVJJ50UDRo0iL59+673fJAGDRrE1VdfHQMHDozu3btH//79ay6126pVq80KqqVLl8Zuu+0Wxx9/fHTu3DkaNWoU06ZNixkzZsSoUaM2+Lju3bvHWWedFSNHjoyZM2fGkUceGQ0aNIjZs2fHxIkT44Ybbojjjz8+GjduHDfddFOceuqp0aVLlzjppJNil112iblz58akSZOiW7dutWKjqKgoHn300RgwYEAceOCBMXny5Jg0aVL89Kc//cJLEG+uM888M377299GZWVlvPDCC9GqVau477774umnn47Ro0fX7FUrLS2Nfv36xZgxYyIvLy/atm0bDz/8cM25LRvjz3/+cwwePDj69esX7dq1i9WrV8ftt98e+fn5cdxxx23x9wbwtcnx1bYAtoj1XWq3qKgo22effbKbbropq66urrX+0qVLs/POOy8rLy/PGjRokO2xxx7ZtddeW7PeCy+8kNWvX7/WpVuzLMtWr16dHXDAAVl5eXm2cOHCDc6z5vKqX3Zp1PVdbjXLsmzatGlZt27dsuLi4qxx48ZZ3759s3/+85/rPH7EiBHZrrvumtWrV2+jLrt77733Zvvuu29WWFiY7bjjjtnJJ5+c/c///E+tdTb2UrsrV67MLrzwwqxz585ZSUlJ1rBhw6xz587Zb37zm1rrff5Su2uMGzcu22+//bLi4uKspKQk69SpU/b//t//y+bNm1drvenTp2c9e/bMSktLs6Kioqxt27ZZZWVl9vzzz9d6jYYNG2ZvvfVWduSRR2Y77LBDVlZWll122WVZVVXVF76Pz+vevXu25557rrN8fe9j/vz52cCBA7Odd945KygoyDp16rTOtsyyLPvwww+z4447Ltthhx2ypk2bZmeddVb2yiuvrPdSuw0bNlzn8W+//XY2aNCgrG3btllRUVG24447Zj169MimTZu2Se8NINfysmwzf/UrANQRlZWVcd9999Wc1wNA3eScDwAAIAnnfACwXfj4449j1apVG7w/Pz//azk3BID/Iz4A2C4ce+yxX/hbxysqKjb5lxECsGmc8wHAduGFF16IhQsXbvD+4uLi6NatW8KJALY/4gMAAEjCCecAAEASm33OR3V1dcybNy9KSkoiLy9vS84EAABsRbIsi6VLl0Z5eXnUq7fh/RubHR/z5s2Lli1bbu7DAQCAbcy7774bu+222wbv3+z4KCkpqXmBxo0bb+7TAAAAW7klS5ZEy5YtaxphQzY7PtYcatW4cWPxAQAAfOnpGE44BwAAkhAfAABAEuIDAABIQnwAAABJiA8AACAJ8QEAACQhPgAAgCTEBwAAkIT4AAAAkhAfAABAEuIDAABI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tb46dOiQ67FYy3vvvRennHJK7LTTTlFcXBydOnWK559/PtdjERGtWrVa5+9PXl5enHPOObkebbtXVVUVl1xySbRu3TqKi4ujbdu2MWLEiHAdnLph6dKlMXTo0KioqIji4uI46KCDYsaMGbkea7v05JNPRt++faO8vDzy8vLigQceqHV/lmVx6aWXRosWLaK4uDgOP/zwmD17dm6G3QjJ4mO33XaLX/ziF/HCCy/E888/H4ceemgcc8wx8Y9//CPVCGyEGTNmxG9/+9vYe++9cz0Ka9lzzz3j/fffr/l66qmncj0S/2vhwoXRrVu3aNCgQUyePDn++c9/xqhRo6Jp06a5Ho34z3/T1v6789hjj0VERL9+/XI8GVdffXXcdNNNceONN8arr74aV199dVxzzTUxZsyYXI9GRJxxxhnx2GOPxe233x6zZs2KI488Mg4//PB47733cj3admf58uXRuXPn+PWvf73e+6+55pr41a9+FWPHjo3nnnsuGjZsGD179oxPP/008aQbJ6eX2t1xxx3j2muvjdNPPz1XI7CWZcuWRZcuXeI3v/lNXHHFFbHPPvvE6NGjcz3Wdm/48OHxwAMPxMyZM3M9Cutx0UUXxdNPPx1/+ctfcj0KG2Ho0KHx8MMPx+zZsyMvLy/X42zXjj766CgrK4ubb765Ztlxxx0XxcXFcccdd+RwMlasWBElJSXx4IMPRp8+fWqW77ffftGrV6+44oorcjjd9i0vLy/uv//++N73vhcR/9nrUV5eHhdccEH8+Mc/joiIxYsXR1lZWYwfPz5OOumkHE67fjk556OqqiruueeeWL58eXTt2jUXI7Ae55xzTvTp0ycOP/zwXI/C58yePTvKy8ujTZs2cfLJJ8fcuXNzPRL/649//GPsv//+0a9fv2jWrFnsu+++8bvf/S7XY7Eeq1atijvuuCMGDRokPOqAgw46KP70pz/FG2+8ERERL7/8cjz11FPRq1evHE/G6tWro6qqKoqKimotLy4utue9jpkzZ0588MEHtT67lZaWxoEHHhjPPvtsDifbsPopX2zWrFnRtWvX+PTTT6NRo0Zx//33xze/+c2UI7AB99xzT7z44ouO56yDDjzwwBg/fny0b98+3n///fj5z38eBx98cLzyyitRUlKS6/G2e2+//XbcdNNNcf7558dPf/rTmDFjRgwZMiQKCgpiwIABuR6PtTzwwAOxaNGiqKyszPUoxH/2Gi5ZsiQ6dOgQ+fn5UVVVFVdeeWWcfPLJuR5tu1dSUhJdu3aNESNGRMeOHaOsrCzuvvvuePbZZ+Mb3/hGrsdjLR988EFERJSVldVaXlZWVnNfXZM0Ptq3bx8zZ86MxYsXx3333RcDBgyIJ554QoDk2LvvvhvnnntuPPbYY+v8lIPcW/ungHvvvXcceOCBUVFREb///e8dslgHVFdXx/777x9XXXVVRETsu+++8corr8TYsWPFRx1z8803R69evaK8vDzXoxARv//97+POO++Mu+66K/bcc8+YOXNmDB06NMrLy/3dqQNuv/32GDRoUOy6666Rn58fXbp0if79+8cLL7yQ69HYyiU97KqgoCC+8Y1vxH777RcjR46Mzp07xw033JByBNbjhRdeiAULFkSXLl2ifv36Ub9+/XjiiSfiV7/6VdSvXz+qqqpyPSJradKkSbRr1y7efPPNXI9CRLRo0WKdH6B07NjRoXF1zDvvvBPTpk2LM844I9ej8L8uvPDCuOiii+Kkk06KTp06xamnnhrnnXdejBw5MtejERFt27aNJ554IpYtWxbvvvtu/O1vf4vPPvss2rRpk+vRWEvz5s0jImL+/Pm1ls+fP7/mvromp7/no7q6OlauXJnLEYiIww47LGbNmhUzZ86s+dp///3j5JNPjpkzZ0Z+fn6uR2Qty5Yti7feeitatGiR61GIiG7dusXrr79ea9kbb7wRFRUVOZqI9bn11lujWbNmtU6eJbc++eSTqFev9seQ/Pz8qK6uztFErE/Dhg2jRYsWsXDhwpgyZUocc8wxuR6JtbRu3TqaN28ef/rTn2qWLVmyJJ577rk6e151ssOuLr744ujVq1fsvvvusXTp0rjrrrvi8ccfjylTpqQagQ0oKSmJvfbaq9ayhg0bxk477bTOctL78Y9/HH379o2KioqYN29eXHbZZZGfnx/9+/fP9WhExHnnnRcHHXRQXHXVVXHCCSfE3/72txg3blyMGzcu16Pxv6qrq+PWW2+NAQMGRP36SY825gv07ds3rrzyyth9991jzz33jJdeeimuu+66GDRoUK5HIyKmTJkSWZZF+/bt480334wLL7wwOnToEAMHDsz1aNudZcuW1TraYc6cOTFz5szYcccdY/fdd4+hQ4fGFVdcEXvssUe0bt06LrnkkigvL6+5IladkyUyaNCgrKKiIisoKMh22WWX7LDDDsumTp2a6uXZRN27d8/OPffcXI9BlmUnnnhi1qJFi6ygoCDbddddsxNPPDF78803cz0Wa3nooYeyvfbaKyssLMw6dOiQjRs3LtcjsZYpU6ZkEZG9/vrruR6FtSxZsiQ799xzs9133z0rKirK2rRpkw0bNixbuXJlrkcjy7J77703a9OmTVZQUJA1b948O+ecc7JFixbleqzt0vTp07OIWOdrwIABWZZlWXV1dXbJJZdkZWVlWWFhYXbYYYfV6f/e5fT3fAAAANuPnJ7zAQAAbD/EBwAAkIT4AAAAkhAfAABAEuIDAABIQnwAAABJiA8AACAJ8QEAACQhPgC2YpWVlfG9730v12PU0qpVqxg9enSuxwCgDhIfAABAEuIDgG3CqlWrcj0CAF9CfABsBe67777o1KlTFBcXx0477RSHH354LF++fJ31qqurY+TIkdG6desoLi6Ozp07x3333VdrnVdeeSV69eoVjRo1irKysjj11FPjo48+qrn/kEMOicGDB8fgwYOjtLQ0dt5557jkkksiy7KNnveTTz6JQYMGRUlJSey+++4xbty4WvfPmjUrDj300Jr3c+aZZ8ayZctqzTB06NBaj/ne974XlZWVNbdbtWoVI0aMiNNOOy0aN24cZ555ZqxatSoGDx4cLVq0iKKioqioqIiRI0du9NwAfL3EB0Ad9/7770f//v1j0KBB8eqrr8bjjz8exx577HpjYOTIkXHbbbfF2LFj4x//+Eecd955ccopp8QTTzwRERGLFi2KQw89NPbdd994/vnn49FHH4358+fHCSecUOt5JkyYEPXr14+//e1vccMNN8R1110X//Vf/7XRM48aNSr233//eOmll+Lss8+OH/7wh/H6669HRMTy5cujZ8+e0bRp05gxY0ZMnDgxpk2bFoMHD97k780vf/nL6Ny5c7z00ktxySWXxK9+9av44x//GL///e/j9ddfjzvvvDNatWq1yc8LwNejfq4HAOCLvf/++7F69eo49thjo6KiIiIiOnXqtM56K1eujKuuuiqmTZsWXbt2jYiINm3axFNPPRW//e1vo3v37nHjjTfGvvvuG1dddVXN42655ZZo2bJlvPHGG9GuXbuIiGjZsmVcf/31kZeXF+3bt49Zs2bF9ddfH9///vc3aubevXvH2WefHRERP/nJT+L666+P6dOnR/v27eOuu+6KTz/9NG677bZo2LBhRETceOON0bdv37j66qujrKxso783hx56aFxwwQU1t+fOnRt77LFHfOc734m8vLya7xcAdYM9HwB1XOfOneOwww6LTp06Rb9+/eJ3v/tdLFy4cJ313nzzzfjkk0/iiCOOiEaNGtV83XbbbfHWW29FRMTLL78c06dPr3V/hw4dIiJq1omI+Pa3vx15eXk1t7t27RqzZ8+OqqqqjZp57733rvn/eXl50bx581iwYEFERLz66qvRuXPnmvCIiOjWrVtUV1fX7B3ZWPvvv3+t25WVlTFz5sxo3759DBkyJKZOnbpJzwfA18ueD4A6Lj8/Px577LF45plnYurUqTFmzJgYNmxYPPfcc7XWW3POxKRJk2LXXXetdV9hYWHNOmv2MHxeixYtttjMDRo0qHU7Ly8vqqurN/rx9erVW+ewss8++2yd9dYOmIiILl26xJw5c2Ly5Mkxbdq0OOGEE+Lwww9f57wXAHJDfABsBfLy8qJbt27RrVu3uPTSS6OioiLuv//+Wut885vfjMLCwpg7d2507959vc/TpUuX+O///u9o1apV1K+/4X8CPh82f/3rX2OPPfaI/Pz8r/xeOnbsGOPHj4/ly5fXxMPTTz8d9erVi/bt20dExC677BLvv/9+zWOqqqrilVdeiR49enzp8zdu3DhOPPHEOPHEE+P444+Po446Kj7++OPYcccdv/LsAHw1DrsCqOOee+65uOqqq+L555+PuXPnxh/+8If48MMPo2PHjrXWKykpiR//+Mdx3nnnxYQJE+Ktt96KF198McaMGRMTJkyIiIhzzjknPv744+jfv3/MmDEj3nrrrZgyZUoMHDiw1iFVc+fOjfPPPz9ef/31uPvuu2PMmDFx7rnnbpH3c/LJJ0dRUVEMGDAgXnnllZg+fXr86Ec/ilNPPbXmfI9DDz00Jk2aFJMmTYrXXnstfvjDH8aiRYu+9Lmvu+66uPvuu+O1116LN954IyZOnBjNmzePJk2abJHZAfhq7PkAqOMaN24cTz75ZIwePTqWLFkSFRUVMWrUqOjVq1fce++9tdYdMWJE7LLLLjFy5Mh4++23o0mTJtGlS5f46U9/GhER5eXl8fTTT8dPfvKTOPLII2PlypVRUVERRx11VNSr938/jzrttNNixYoV8a1vfSvy8/Pj3HPPjTPPPHOLvJ8ddtghpkyZEueee24ccMABscMOO8Rxxx0X1113Xc06gwYNipdffjlOO+20qF+/fpx33nkbtdejpKQkrrnmmpg9e3bk5+fHAQccEI888kit9wZA7uRlm3LhdgC2eYccckjss88+MXr06FyPAsA2xo+CAACAJMQHABvtL3/5S63L9H7+CwC+iMOuANhoK1asiPfee2+D93/jG99IOA0AWxvxAQAAJOGwKwAAIAnxAQAAJCE+AACAJMQHAACQhPgAAACSEB8AAEAS4gMAAEji/wceg2UpeZVorwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAMaRJREFUeJzt3Xu8lXPe+P/3brfbO7XbRbV1Usk5UtQ0xIQQqsEgupkOzlQozKMevpIxI26DGkM0c6umyYxjxjCJHGcwQ9zlTIiGqEipqGhfvz/8Wret0+7gs5Tn8/HoMe1rXde63uta27Ree63r2gVZlmUBAADwHauW7wEAAIAfBvEBAAAkIT4AAIAkxAcAAJCE+AAAAJIQHwAAQBLiAwAASEJ8AAAASYgPAAAgCfEB/OC9++67UVBQEOPGjcv3KJU8+OCD0bZt2ygpKYmCgoJYuHBhvkfabL4Px3zu3Llx/PHHx3bbbRcFBQUxcuTIvM0C8EMhPoDNZty4cVFQUFDpT8OGDePggw+OyZMnJ5/n8ccfrzRLUVFR7LjjjtG7d+945513Nss+nn766Rg+fPhmD4NPPvkkevbsGTVr1owbb7wxJkyYELVq1dqs+/ihGzRoUEyZMiWGDh0aEyZMiCOOOCLfIwFs9arnewBg6/PLX/4yWrZsGVmWxdy5c2PcuHFx1FFHxd/+9rfo3r178nnOO++86NChQ3z55ZfxwgsvxJgxY+KBBx6Il156KRo3brxJ9/3000/H5ZdfHn379o26detunoEj4rnnnovFixfHFVdcEYceeuhmu9/vi+bNm8cXX3wRRUVFeZvh0UcfjaOPPjouuuiivM0A8EMjPoDN7sgjj4z27dvnvj7ttNOivLw8/vznP+clPg488MA4/vjjIyKiX79+scsuu8R5550X48ePj6FDhyafpyrmzZsXEbFZg+a7lmVZLFu2LGrWrLnedQsKCqKkpCTBVGs3b968Kh3fpUuXetcJYDPxsSvgO1e3bt2oWbNmVK9e+ecdS5cujQsvvDCaNWsWxcXFseuuu8ZvfvObyLIsIiK++OKL2G233WK33XaLL774IrfdggULolGjRrH//vvHypUrN3ieQw45JCIiZs2atc71Hn300TjwwAOjVq1aUbdu3Tj66KPjtddey90+fPjwuPjiiyMiomXLlrmPd7377rvrvN8777wz9t1336hZs2bUr18/TjnllPjggw9ytx900EHRp0+fiIjo0KFDFBQURN++fdd5nx988EGceuqpUV5eHsXFxdG6deu49dZbc7dvyLGsqKiIkSNHRuvWraOkpCTKy8vjrLPOik8//bTSPlu0aBHdu3ePKVOmRPv27aNmzZpxyy23RETEwoULY9CgQdGiRYsoLi6Opk2bRu/evePjjz+OiDWf8/HRRx9Fv379omnTplFcXByNGjWKo48+erXjOXny5NzzUlpaGt26dYtXXnllncfnm1Z9PDDLsrjxxhtzz9s3b3viiSfi3HPPjYYNG0bTpk03eN/33ntv7LnnnlFSUhJ77rlnTJo0Kfr27RstWrTIrbPqY4GPP/54pW3Xdj7M66+/Hscff3xsu+22UVJSEu3bt4/77rtvjY/tqaeeisGDB0eDBg2iVq1aceyxx8b8+fNXm3Py5MnRuXPnKC0tjTp16kSHDh3itttui4iIyy67LIqKita43Zlnnhl169aNZcuWrfNYA3yb+AA2u0WLFsXHH38c8+fPj1deeSXOOeecWLJkSZxyyim5dbIsi5/+9Kdx/fXXxxFHHBHXXXdd7LrrrnHxxRfH4MGDIyKiZs2aMX78+HjrrbfikksuyW3bv3//WLRoUYwbNy4KCws3eL633347IiK22267ta4zderU6Nq1a8ybNy+GDx8egwcPjqeffjo6deqUezH8s5/9LHr16hUREddff31MmDAhJkyYEA0aNFjr/Y4bNy569uwZhYWFMWLEiDjjjDPinnvuiQMOOCB33sgll1wSZ555ZkR8/RG2CRMmxFlnnbXW+5w7d278+Mc/jqlTp8aAAQNi1KhRsdNOO8Vpp52WO4l6Q47lWWedFRdffHF06tQpRo0aFf369YuJEydG165d48svv6y07zfeeCN69eoVhx12WIwaNSratm0bS5YsiQMPPDBuuOGGOPzww2PUqFFx9tlnx+uvvx7vv//+Wh/HcccdF5MmTYp+/frFTTfdFOedd14sXrw4Zs+enVtnwoQJ0a1bt6hdu3ZcffXVcemll8arr74aBxxwwHqjb5Wf/OQnMWHChIiIOOyww3LP2zede+658eqrr8awYcNiyJAhG7Tvhx56KI477rgoKCiIESNGxDHHHBP9+vWLadOmVWm+NXnllVfixz/+cbz22msxZMiQuPbaa6NWrVpxzDHHxKRJk1Zbf+DAgTFjxoy47LLL4pxzzom//e1vMWDAgErrjBs3Lrp16xYLFiyIoUOHxlVXXRVt27aNBx98MCIifv7zn8dXX30Vt99+e6XtVqxYEXfddVccd9xxeX/3CtgCZQCbydixY7OIWO1PcXFxNm7cuErr3nvvvVlEZL/61a8qLT/++OOzgoKC7K233sotGzp0aFatWrXsySefzO68884sIrKRI0eud57HHnssi4js1ltvzebPn5/NmTMne+CBB7IWLVpkBQUF2XPPPZdlWZbNmjUri4hs7NixuW3btm2bNWzYMPvkk09yy2bMmJFVq1Yt6927d27ZNddck0VENmvWrPXOs2LFiqxhw4bZnnvumX3xxRe55ffff38WEdmwYcNyy1Ydy1Uzrstpp52WNWrUKPv4448rLT/ppJOysrKy7PPPP88tW9+x/Mc//pFFRDZx4sRK9/Xggw+utrx58+ZZRGQPPvhgpXWHDRuWRUR2zz33rDZrRUVFlmWrH/NPP/00i4jsmmuuWevjXLx4cVa3bt3sjDPOqLT8o48+ysrKylZbvj4RkfXv37/SslXH/YADDsi++uqrjdp327Zts0aNGmULFy7MLXvooYeyiMiaN2+eW7bq+/Oxxx6rdJ9r+n7s0qVLttdee2XLli3LLauoqMj233//bOedd15t/kMPPTR3rLMsywYNGpQVFhbmZlq4cGFWWlqadezYsdL34qr7XWW//fbLOnbsWOn2e+65Z41zA1SFdz6Aze7GG2+Mhx9+OB5++OH405/+FAcffHCcfvrpcc899+TW+fvf/x6FhYVx3nnnVdr2wgsvjCzLKl0da/jw4dG6devo06dPnHvuudG5c+fVtluXU089NRo0aBCNGzeObt26xdKlS2P8+PGVzkv5pg8//DCmT58effv2jW233Ta3vE2bNnHYYYfF3//+9yrv+5umTZsW8+bNi3PPPbfST4y7desWu+22WzzwwAMbfJ9ZlsXdd98dPXr0iCzL4uOPP8796dq1ayxatCheeOGF3PrrO5Z33nlnlJWVxWGHHVbpvvbdd9+oXbt2PPbYY5X237Jly+jatWulZXfffXfsvffeceyxx64276qPN31bzZo1o0aNGvH444+v9vGuVR5++OFYuHBh9OrVq9JshYWF0bFjx9Vm2xRnnHFGpXfVqrrvVd87ffr0ibKystz2hx12WOyxxx4bNcuCBQvi0UcfjZ49e8bixYtz+/7kk0+ia9euMXPmzEof24v4+mNR3zzWBx54YKxcuTLee++93ONZvHhxDBkyZLV3L765Xe/evePf//537t3CiIiJEydGs2bNonPnzhv1eIAfNiecA5vdj370o0ov7Hv16hXt2rWLAQMGRPfu3aNGjRrx3nvvRePGjaO0tLTStrvvvntERO5FUkREjRo14tZbb40OHTpESUlJjB07dq0vYtdk2LBhceCBB0ZhYWHUr18/dt9999XOP/mmVfveddddV7tt9913jylTpmzUScjrut/ddtst/vnPf27Q/UVEzJ8/PxYuXBhjxoyJMWPGrHGdVSevR6z/WM6cOTMWLVoUDRs2XO99RXwdH9/29ttvx3HHHbdBj6O4uDiuvvrquPDCC6O8vDx+/OMfR/fu3aN3796x/fbb52aL+L9zdr6tTp06G7TPdfn246rqvlc9xzvvvPNq6+y6666VQrCq3nrrrciyLC699NK49NJL17jOvHnzokmTJrmvd9hhh0q316tXLyIiF3arYmLPPfdc575PPPHEuOCCC2LixIkxbNiwWLRoUdx///0xaNCgDfpvEGAV8QF856pVqxYHH3xwjBo1KmbOnBmtW7fe4PuYMmVKREQsW7YsZs6cucYXvWuz1157bZWXq434+uTwiIhTTjkld5L6t7Vp06bS1+s6lhUVFdGwYcOYOHHiGu/r2+ezVOXKVlV1wQUXRI8ePeLee++NKVOmxKWXXhojRoyIRx99NNq1a5d7rBMmTMgFyTetKyg31Lcf13ex77W9eP/2RRRW7fuiiy5a7V2mVXbaaadKX6/tXKjs/7+YQ1XVq1cvunfvnouPu+66K5YvX17p/C2ADSE+gCS++uqriIhYsmRJRHz9ex6mTp0aixcvrvTux+uvv567fZUXX3wxfvnLX0a/fv1i+vTpcfrpp8dLL71U6WMtm9Oqfb/xxhur3fb6669H/fr1c+96bMhPf795v9/+Cfobb7xR6TFXVYMGDaK0tDRWrlxZpcBa37Fs1apVTJ06NTp16rTRYdGqVat4+eWXN3rbCy+8MC688MKYOXNmtG3bNq699tr405/+FK1atYqIiIYNGyaPyarue9VzuOqdkm/69vfTqncjvv0LKr/5rl9ExI477hgREUVFRZvtca96PC+//PJq4fJtvXv3jqOPPjqee+65mDhxYrRr126jfoAAEOFqV0ACX375ZTz00ENRo0aN3MeqjjrqqFi5cmX87ne/q7Tu9ddfHwUFBXHkkUfmtu3bt280btw4Ro0aFePGjYu5c+fGoEGDvrN5GzVqFG3bto3x48dXemH48ssvx0MPPRRHHXVUbtmqCKnKbzhv3759NGzYMG6++eZYvnx5bvnkyZPjtddei27dum3wrIWFhXHcccfF3XffvcYX/N+8TGpVjmXPnj1j5cqVccUVV6x2X1999VWVHudxxx0XM2bMWONVmNb2k/fPP/98tcu2tmrVKkpLS3PHqmvXrlGnTp248sorV7vq1rcf6+ZW1X1/83tn0aJFudsffvjhePXVVytt07x58ygsLIwnn3yy0vKbbrqp0tcNGzaMgw46KG655Zb48MMP17rvDXH44YdHaWlpjBgxYrXj/u3n6Mgjj4z69evH1VdfHU888YR3PYBN4p0PYLObPHly7h2MefPmxW233RYzZ86MIUOG5D4b36NHjzj44IPjkksuiXfffTf23nvveOihh+Kvf/1rXHDBBbmfzP7qV7+K6dOnxyOPPBKlpaXRpk2bGDZsWPy///f/4vjjj68UApvTNddcE0ceeWTst99+cdppp8UXX3wRN9xwQ5SVlcXw4cNz6+27774R8fXlcU866aQoKiqKHj16rPF8kKKiorj66qujX79+0blz5+jVq1fMnTs3Ro0aFS1atNjooLrqqqvisccei44dO8YZZ5wRe+yxRyxYsCBeeOGFmDp1aixYsCAiqnYsO3fuHGeddVaMGDEipk+fHocffngUFRXFzJkz484774xRo0blfmHj2lx88cVx1113xQknnBCnnnpq7LvvvrFgwYK477774uabb4699957tW3efPPN6NKlS/Ts2TP22GOPqF69ekyaNCnmzp0bJ510UkR8fV7F6NGj4+c//3nss88+cdJJJ0WDBg1i9uzZ8cADD0SnTp1Wi9nNZUP2PWLEiOjWrVsccMABceqpp8aCBQvihhtuiNatW+fe+YuIKCsrixNOOCFuuOGGKCgoiFatWsX999+/2nk1EV9fxOGAAw6IvfbaK84444zYcccdY+7cufHMM8/E+++/HzNmzNjgx3P99dfH6aefHh06dIj/+q//inr16sWMGTPi888/j/Hjx+fWLSoqipNOOil+97vfRWFhYe7y0gAbJY9X2gK2Mmu61G5JSUnWtm3bbPTo0ZUu4ZllX1++dNCgQVnjxo2zoqKibOedd86uueaa3HrPP/98Vr169WzgwIGVtvvqq6+yDh06ZI0bN84+/fTTtc6z6lKmd9555zrnXtOlTbMsy6ZOnZp16tQpq1mzZlanTp2sR48e2auvvrra9ldccUXWpEmTrFq1alW67O7tt9+etWvXLisuLs623Xbb7OSTT87ef//9SutsyKV2syzL5s6dm/Xv3z9r1qxZVlRUlG2//fZZly5dsjFjxmRZtuHHcsyYMdm+++6b1axZMystLc322muv7Be/+EU2Z86c3DrNmzfPunXrtsZ5Pvnkk2zAgAFZkyZNsho1amRNmzbN+vTpk7sc8LeP+ccff5z1798/22233bJatWplZWVlWceOHbM77rhjtft+7LHHsq5du2ZlZWVZSUlJ1qpVq6xv377ZtGnTqnSsVol1XGp3bce9qvu+++67s9133z0rLi7O9thjj+yee+7J+vTpU+lSu1mWZfPnz8+OO+64bJtttsnq1auXnXXWWdnLL7+8xu/Ht99+O+vdu3e2/fbbZ0VFRVmTJk2y7t27Z3fdddd651/bZX3vu+++bP/99899j//oRz/K/vznP6/2uJ999tksIrLDDz98jccFoKoKsmwDzz4DADZY37594/HHH6/yL0P8PpkxY0a0bds2/vjHP8bPf/7zfI8DbMGc8wEArNPvf//7qF27dvzsZz/L9yjAFs45HwBsNVasWJE7x2VtysrKNuslgrdmf/vb3+LVV1+NMWPGxIABAzb4d9sAfJv4AGCr8fTTT8fBBx+8znXGjh0bffv2TTPQFm7gwIExd+7cOOqoo+Lyyy/P9zjAVsA5HwBsNT799NN4/vnn17lO69ato1GjRokmAuCbxAcAAJCEE84BAIAkNvqcj4qKipgzZ06UlpZGQUHB5pwJAADYgmRZFosXL47GjRtHtWprf39jo+Njzpw50axZs43dHAAA2Mr85z//iaZNm6719o2Oj9LS0twO6tSps7F3AwAAbOE+++yzaNasWa4R1maj42PVR63q1KkjPgAAgPWejuGEcwAAIAnxAQAAJCE+AACAJMQHAACQhPgAAACSEB8AAEAS4gMAAEhCfAAAAEmIDwAAIAnxAQAAJCE+AACAJMQHAACQhPgAAACSEB8A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kybFMccck+9R2Izmz58fDRs2jCeeeCJ+8pOf5HscNpNtt902rrnmmjjttNPyPQqbYMmSJbHPPvvETTfdFL/61a+ibdu2MXLkyHyPxUYaPnx43HvvvTF9+vR8j8JmMmTIkHjqqafiH//4R75H2Shb9DsfK1asiOeffz4OPfTQ3LJq1arFoYceGs8880weJwPWZdGiRRHx9YtVtnwrV66Mv/zlL7F06dLYb7/98j0Om6h///7RrVu3Sv+2smWbOXNmNG7cOHbcccc4+eSTY/bs2fkeiU1w3333Rfv27eOEE06Ihg0bRrt27eL3v/99vseqsi06Pj7++ONYuXJllJeXV1peXl4eH330UZ6mAtaloqIiLrjggujUqVPsueee+R6HTfDSSy9F7dq1o7i4OM4+++yYNGlS7LHHHvkei03wl7/8JV544YUYMWJEvkdhM+nYsWOMGzcuHnzwwRg9enTMmjUrDjzwwFi8eHG+R2MjvfPOOzF69OjYeeedY8qUKXHOOefEeeedF+PHj8/3aFVSPd8DAD8s/fv3j5dfftlnjrcCu+66a0yfPj0WLVoUd911V/Tp0yeeeOIJAbKF+s9//hPnn39+PPzww1FSUpLvcdhMjjzyyNzf27RpEx07dozmzZvHHXfc4SOSW6iKiopo3759XHnllRER0a5du3j55Zfj5ptvjj59+uR5uvXbot/5qF+/fhQWFsbcuXMrLZ87d25sv/32eZoKWJsBAwbE/fffH4899lg0bdo03+OwiWrUqBE77bRT7LvvvjFixIjYe++9Y9SoUfkei430/PPPx7x582KfffaJ6tWrR/Xq1eOJJ56I3/72t1G9evVYuXJlvkdkM6hbt27ssssu8dZbb+V7FDZSo0aNVvshz+67777FfJxui46PGjVqxL777huPPPJIbllFRUU88sgjPncM3yNZlsWAAQNi0qRJ8eijj0bLli3zPRLfgYqKili+fHm+x2AjdenSJV566aWYPn167k/79u3j5JNPjunTp0dhYWG+R2QzWLJkSbz99tvRqFGjfI/CRurUqdNql6t/8803o3nz5nmaaMNs8R+7Gjx4cPTp0yfat28fP/rRj2LkyJGxdOnS6NevX75HYyMtWbKk0k9kZs2aFdOnT49tt902dthhhzxOxsbq379/3HbbbfHXv/41SktLc+dklZWVRc2aNfM8HRtj6NChceSRR8YOO+wQixcvjttuuy0ef/zxmDJlSr5HYyOVlpaudh5WrVq1YrvttnN+1hbsoosuih49ekTz5s1jzpw5cdlll0VhYWH06tUr36OxkQYNGhT7779/XHnlldGzZ8949tlnY8yYMTFmzJh8j1YlW3x8nHjiiTF//vwYNmxYfPTRR9G2bdt48MEHVzsJnS3HtGnT4uCDD859PXjw4IiI6NOnT4wbNy5PU7EpRo8eHRERBx10UKXlY8eOjb59+6YfiE02b9686N27d3z44YdRVlYWbdq0iSlTpsRhhx2W79GAb3j//fejV69e8cknn0SDBg3igAMOiH/961/RoEGDfI/GRurQoUNMmjQphg4dGr/85S+jZcuWMXLkyDj55JPzPVqVbPG/5wMAANgybNHnfAAAAFsO8QEAACQhPgAAgCTEBwAAkIT4AAAAkhAfAABAEuIDAABIQnwAAABJiA+ArUxBQUHce++9SfY1ZsyYaNasWVSrVi1GjhyZZJ8AbLn8hnOArcxHH30U9erVi+Li4u90P5999lnUr18/rrvuujjuuOOirKwsttlmm+90nwBs2cQHwBbkyy+/jKKionyPERERL7/8cuy1117xzjvvRMuWLde4zooVK6JGjRqJJwPg+8rHrgA2QUVFRYwYMSJatmwZNWvWjL333jvuuuuuyLIsDj300OjatWus+hnPggULomnTpjFs2LDc9n/4wx9i9913j5KSkthtt93ipptuyt327rvvRkFBQdx+++3RuXPnKCkpiYkTJ0ZExK233hqtW7eO4uLiaNSoUQwYMCC33Tc/drVixYoYMGBANGrUKEpKSqJ58+YxYsSI3LoLFy6M008/PRo0aBB16tSJQw45JGbMmLHexz1u3LjYa6+9IiJixx13jIKCgnj33Xdj+PDh0bZt2/jDH/4QLVu2jJKSkirv56qrrory8vIoLS2N0047LYYMGRJt27bN3X7QQQfFBRdcUGmbY445Jvr27Zv7evny5XHRRRdFkyZNolatWtGxY8d4/PHHK81dt27dmDJlSuy+++5Ru3btOOKII+LDDz+sdL9rO76nnnpqdO/evdK6X375ZTRs2DD+53/+Z73HDeCHTnwAbIIRI0bEH//4x7j55pvjlVdeiUGDBsUpp5wSTz75ZIwfPz6ee+65+O1vfxsREWeffXY0adIkFx8TJ06MYcOGxa9//et47bXX4sorr4xLL700xo8fX2kfQ4YMifPPPz9ee+216Nq1a4wePTr69+8fZ555Zrz00ktx3333xU477bTG+X7729/GfffdF3fccUe88cYbMXHixGjRokXu9hNOOCHmzZsXkydPjueffz722Wef6NKlSyxYsGCdj/vEE0+MqVOnRkTEs88+Gx9++GE0a9YsIiLeeuutuPvuu+Oee+6J6dOnV2k/d9xxRwwfPjyuvPLKmDZtWjRq1KhSiFXVgAED4plnnom//OUv8eKLL8YJJ5wQRxxxRMycOTO3zueffx6/+c1vYsKECfHkk0/G7Nmz46KLLsrdvq7je/rpp8eDDz5YKVbuv//++Pzzz+PEE0/c4HkBfnAyADbKsmXLsm222SZ7+umnKy0/7bTTsl69emVZlmV33HFHVlJSkg0ZMiSrVatW9uabb+bWa9WqVXbbbbdV2vaKK67I9ttvvyzLsmzWrFlZRGQjR46stE7jxo2zSy65ZK1zRUQ2adKkLMuybODAgdkhhxySVVRUrLbeP/7xj6xOnTrZsmXLKi1v1apVdsstt6zn0WfZ//7v/2YRkc2aNSu37LLLLsuKioqyefPmbdB+9ttvv+zcc8+tdHvHjh2zvffeO/d1586ds/PPP7/SOkcffXTWp0+fLMuy7L333ssKCwuzDz74oNI6Xbp0yYYOHZplWZaNHTs2i4jsrbfeyt1+4403ZuXl5bmv13d899hjj+zqq6/Ofd2jR4+sb9++a10fgP9TPb/pA7Dleuutt+Lzzz+Pww47rNLyFStWRLt27SLi65/4T5o0Ka666qoYPXp07LzzzhERsXTp0nj77bfjtNNOizPOOCO37VdffRVlZWWV7q99+/a5v8+bNy/mzJkTXbp0qdKMffv2jcMOOyx23XXXOOKII6J79+5x+OGHR0TEjBkzYsmSJbHddttV2uaLL76It99+u4pHYXXNmzePBg0a5L6uyn5ee+21OPvssyvdvt9++8Vjjz1W5f2+9NJLsXLlythll10qLV++fHmlfW+zzTbRqlWr3NeNGjWKefPmRUTVju/pp58eY8aMiV/84hcxd+7cmDx5cjz66KNVnhPgh0x8AGykJUuWRETEAw88EE2aNKl026orTX3++efx/PPPR2FhYaWP/qza9ve//3107Nix0raFhYWVvq5Vq1bu7zVr1tygGffZZ5+YNWtWTJ48OaZOnRo9e/aMQw89NO66665YsmRJNGrUqNI5EavUrVt3g/aztnkjYrPtp1q1arnzZ1b58ssvK+2nsLAwd7y/qXbt2rm/f/uE/YKCgtz9VuX49u7dO4YMGRLPPPNMPP3009GyZcs48MADq/w4AH7IxAfARtpjjz2iuLg4Zs+eHZ07d17jOhdeeGFUq1YtJk+eHEcddVR069YtDjnkkCgvL4/GjRvHO++8EyeffHKV91laWhotWrSIRx55JA4++OAqbVOnTp048cQT48QTT4zjjz8+jjjiiFiwYEHss88+8dFHH0X16tUrnQeyuVVlP7vvvnv8+9//jt69e+eW/etf/6q0ToMGDSqda7Fy5cp4+eWXc8ehXbt2sXLlypg3b95Gx0BVju92220XxxxzTIwdOzaeeeaZ6Nev30btC+CHSHwAbKTS0tK46KKLYtCgQVFRUREHHHBALFq0KJ566qmoU6dO1K9fP2699dZ45plnYp999omLL744+vTpEy+++GLUq1cvLr/88jjvvPOirKwsjjjiiFi+fHlMmzYtPv300xg8ePBa9zt8+PA4++yzo2HDhnHkkUfG4sWL46mnnoqBAweutu51110XjRo1inbt2kW1atXizjvvjO233z7q1q0bhx56aOy3335xzDHHxH//93/HLrvsEnPmzIkHHnggjj322Eof99oUVdnP+eefH3379o327dtHp06dYuLEifHKK6/EjjvumLufQw45JAYPHhwPPPBAtGrVKq677rpYuHBh7vZddtklTj755Ojdu3dce+210a5du5g/f3488sgj0aZNm+jWrVuV5q3K8T399NOje/fusXLlyujTp89mOU4APwj5PukEYEtWUVGRjRw5Mtt1112zoqKirEGDBlnXrl2zxx9/PCsvL8+uvPLK3LorVqzI9t1336xnz565ZRMnTszatm2b1ahRI6tXr172k5/8JLvnnnuyLPu/E87/93//d7X93nzzzbl9NmrUKBs4cGDutvjGCedjxozJ2rZtm9WqVSurU6dO1qVLl+yFF17IrfvZZ59lAwcOzBo3bpwVFRVlzZo1y04++eRs9uzZ633sazvh/JsniW/Ifn79619n9evXz2rXrp316dMn+8UvflHpvlasWJGdc8452bbbbps1bNgwGzFiRKUTzletM2zYsKxFixa5Y3PsscdmL774YpZlX59wXlZWVmm2SZMmZd/+53BdxzfLvn7emzdvnh111FHrPU4A/B+/ZBCA76Xhw4fHvffem7tc7/fJkiVLokmTJjF27Nj42c9+lu9xALYYPnYFAFVUUVERH3/8cVx77bVRt27d+OlPf5rvkQC2KH7JIABr1Lp166hdu/Ya/6z6Tes/NLNnz47y8vK47bbb4tZbb43q1f0MD2BD+NgVAGv03nvvVbqU7TeVl5dHaWlp4okA2NKJDwAAIAkfuwIAAJIQHwAAQBLiAwAASEJ8AAAASYgPAAAgCfEBAAAkIT4AAIAk/j/ycxh9MVZufgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAALZpJREFUeJzt3Xu8lXPe+P/3ale7PVvnlPZ0JCphouKhpDIOQ2LM0IwJ5TDmvoU7xF0MQsY0ToMxmbqpJoxhDGOcGhFG4zY5JG5UKAypJLuDiPb1/WN+rV9bp13qs7R7Ph+P/dC61lrXel+rNdN6rXVd185lWZYFAADAVlaj0AMAAADbB/EBAAAkIT4AAIAkxAcAAJCE+AAAAJIQHwAAQBLiAwAASEJ8AAAASYgPAAAgCfEBVHtz586NXC4X48ePL/QolTz66KPRuXPnqFOnTuRyufjkk08KPVJSvXv3jt69e2/Sfdq0aRNHHnnk1hmoCo89aNCgKt+2UHNuivHjx0cul4u5c+cWehRgOyE+gCpb/UZlzZ+mTZtGnz594pFHHkk+z5NPPllpllq1asXOO+8cJ510Urz99ttb5DH+8Y9/xIgRI7Z4GCxatCj69+8fJSUlcfPNN8fEiROjtLR0iz7G1rS1npdtyWuvvRYjRozYJt64/+IXv4j777+/0GMAiA9g011++eUxceLE+P3vfx8XXHBBLFy4MI444oh48MEHCzLP2WefHRMnTowxY8ZE3759449//GN069YtPvjgg6+97n/84x9x2WWXbfE32dOmTYulS5fGFVdcEaeeemqccMIJUatWrS36GFvT1npetiWvvfZaXHbZZdt0fJx44omxYsWKaN26dfqhgO1SzUIPAGx7Dj/88OjatWv+8qmnnhrNmjWLP/zhDwXZ1aRnz55x7LHHRkTEySefHLvttlucffbZMWHChBg+fHjyeapiwYIFERHRoEGDwg7CNqeioiJWrlwZderU+drrKioqiqKioi0wFUDV+OYD+NoaNGgQJSUlUbNm5c8zli9fHuedd160bNkyiouLo3379nHNNddElmUREbFixYro0KFDdOjQIVasWJG/38cffxzNmzeP7t27x6pVqzZ5noMOOigiIubMmbPB2z3xxBPRs2fPKC0tjQYNGsTRRx8dr7/+ev76ESNGxPnnnx8REW3bts3v3rWxT7rvueee6NKlS5SUlESTJk3ihBNOiPfffz9/fe/evWPgwIEREdGtW7fI5XIbPJZgxIgRkcvlYtasWXHCCSdE/fr1Y8cdd4yLL744siyL9957L44++uioV69e7LTTTnHttdeutY7PP/88Lr300mjXrl0UFxdHy5Yt44ILLojPP/+80u1yuVyceeaZcf/998cee+wRxcXF0alTp3j00Uer/LyMGzcuDjrooGjatGkUFxfH7rvvHqNHj97gc7apnnnmmdh3332jTp06sfPOO8fvf//7tW7zySefxJAhQ/Kvv3bt2sWoUaOioqKi0u2uueaa6N69ezRu3DhKSkqiS5cu8ac//WmDjz9+/Pg47rjjIiKiT58++efgySef3OQ5N2b138kdd9wRnTp1iuLi4vzfR1Vmz+VysXz58pgwYUJ+ztWvt3Ud87H6eJWqzD5jxozo1atXlJSURIsWLWLkyJExbtw4x5EA6+WbD2CTlZeXx0cffRRZlsWCBQvipptuimXLlsUJJ5yQv02WZXHUUUfFlClT4tRTT43OnTvHpEmT4vzzz4/3338/rr/++igpKYkJEyZEjx494qKLLorrrrsuIiIGDx4c5eXlMX78+M36VPatt96KiIjGjRuv9zaTJ0+Oww8/PHbeeecYMWJErFixIm666abo0aNHvPjii9GmTZv4wQ9+ELNmzYo//OEPcf3110eTJk0iImLHHXdc73rHjx8fJ598cnTr1i2uuuqqmD9/ftxwww0xderUeOmll6JBgwZx0UUXRfv27WPMmDFx+eWXR9u2bWOXXXbZ6Hb96Ec/io4dO8Yvf/nLeOihh2LkyJHRqFGj+N3vfhcHHXRQjBo1Ku64444YOnRodOvWLQ488MCI+Pcn5UcddVQ888wzcfrpp0fHjh3jlVdeieuvvz5mzZq11u44zzzzTPz5z3+OM844I+rWrRs33nhj/PCHP4x33303GjduvNHnZfTo0dGpU6c46qijombNmvHXv/41zjjjjKioqIjBgwdvdDs35s0334xjjz02Tj311Bg4cGDcdtttMWjQoOjSpUt06tQpIiI+/fTT6NWrV7z//vvxs5/9LFq1ahX/+Mc/Yvjw4TFv3rz49a9/nV/fDTfcEEcddVQMGDAgVq5cGXfddVccd9xx8eCDD0bfvn3XOcOBBx4YZ599dtx4441x4YUXRseOHSMi8v+t6pxV9cQTT8Tdd98dZ555ZjRp0iTatGlT5dknTpwYp512Wuy7775x+umnR0Rs9PVWldnff//9fHgNHz48SktL43/+53+iuLh4k7YN2M5kAFU0bty4LCLW+ikuLs7Gjx9f6bb3339/FhHZyJEjKy0/9thjs1wul7355pv5ZcOHD89q1KiRPf3009k999yTRUT261//eqPzTJkyJYuI7LbbbssWLlyYffDBB9lDDz2UtWnTJsvlctm0adOyLMuyOXPmZBGRjRs3Ln/fzp07Z02bNs0WLVqUX/byyy9nNWrUyE466aT8squvvjqLiGzOnDkbnWflypVZ06ZNsz322CNbsWJFfvmDDz6YRUR2ySWX5Jetfi5Xz7ghl156aRYR2emnn55f9uWXX2YtWrTIcrlc9stf/jK/fPHixVlJSUk2cODA/LKJEydmNWrUyP7+979XWu8tt9ySRUQ2derU/LKIyGrXrl3p7+fll1/OIiK76aab8ss29Lx8+umnay077LDDsp133rnSsl69emW9evXa6PavqXXr1llEZE8//XR+2YIFC7Li4uLsvPPOyy+74oorstLS0mzWrFmV7j9s2LCsqKgoe/fdd9c778qVK7M99tgjO+igg9Z67DWf19Wv1SlTpmz2nFUREVmNGjWy//u//1vruqrOXlpaWmn21Va/Dtf8e6zq7GeddVaWy+Wyl156Kb9s0aJFWaNGjar8vxlg+2O3K2CT3XzzzfHYY4/FY489Frfffnv06dMnTjvttPjzn/+cv83DDz8cRUVFcfbZZ1e673nnnRdZllU6O9aIESOiU6dOMXDgwDjjjDOiV69ea91vQ0455ZTYcccdo6ysLPr27ZvfxWTN41LWNG/evJg+fXoMGjQoGjVqlF++1157xSGHHBIPP/xwlR97Tc8//3wsWLAgzjjjjEr74/ft2zc6dOgQDz300Gatd7XTTjst/+eioqLo2rVrZFkWp556an55gwYNon379pXO9nXPPfdEx44do0OHDvHRRx/lf1bvnjZlypRKj3PwwQdX+mR8r732inr16lX5DGIlJSX5P6/+lqxXr17x9ttvR3l5+aZt9Drsvvvu0bNnz/zlHXfccZ3b3LNnz2jYsGGlbT744INj1apV8fTTT69z3sWLF0d5eXn07NkzXnzxxa0+Z1X16tUrdt9997WWF3L2Rx99NPbff//o3LlzflmjRo1iwIABX+uxgerNblfAJtt3330rvbE//vjjY++9944zzzwzjjzyyKhdu3a88847UVZWFnXr1q1039W7pbzzzjv5ZbVr147bbrstunXrFnXq1MnvM15Vl1xySfTs2TOKioqiSZMm0bFjx7WOP1nT6sdu3779Wtd17NgxJk2aFMuXL9/kU99uaL0dOnSIZ555ZpPW91WtWrWqdLl+/fpRp06d/G5Pay5ftGhR/vLs2bPj9ddfX+/uYqsPfl/f40RENGzYMBYvXlylOadOnRqXXnppPPvss/Hpp59Wuq68vDzq169fpfWsT1Xmmz17dsyYMaNK2/zggw/GyJEjY/r06ZWOgdmU1+DmzllVbdu2XefyQs7+zjvvxP7777/W7dq1a/e1Hhuo3sQH8LXVqFEj+vTpEzfccEPMnj17k/dnj4iYNGlSRER89tlnMXv27PW+2VqXPffcMw4++OBNfsxtzbqOf1nfMTHZ/3dQf8S/j/nYc88988fUfFXLli03eZ3r89Zbb8V3v/vd6NChQ1x33XXRsmXLqF27djz88MNx/fXXr3Ww9+ao6jYfcsghccEFF6zztrvttltERPz973+Po446Kg488MD47W9/G82bN49atWrFuHHj4s4779zqc1bVmt9wrLatzA6wJvEBbBFffvllREQsW7YsIiJat24dkydPjqVLl1b69uONN97IX7/ajBkz4vLLL4+TTz45pk+fHqeddlq88sorX/sT8vVZ/dgzZ85c67o33ngjmjRpkv/WY1M+QV5zvat3aVpt5syZBftdCrvssku8/PLL8d3vfvdrfyK+2vrW89e//jU+//zzeOCBByp9ev7VXbu2tl122SWWLVu20Si99957o06dOjFp0qRKB0qPGzduo4+xpZ7LzbUps2+NWVu3bh1vvvnmWsvXtQxgNcd8AF/bF198EX/729+idu3a+d2qjjjiiFi1alX85je/qXTb66+/PnK5XBx++OH5+w4aNCjKysrihhtuiPHjx8f8+fPjnHPO2WrzNm/ePDp37hwTJkyo9EvyXn311fjb3/4WRxxxRH7Z6gipyi/T69q1azRt2jRuueWWSrvAPPLII/H666+v98xJW1v//v3j/fffj7Fjx6513YoVK2L58uWbvM71PS+rPzFf8xPy8vLyKr2Z35L69+8fzz77bP4btTV98skn+VguKiqK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jtS85kjJ643UtsXX3GYfcA4AALApfPMBAAAkIT4AAIAkxAcAAJCE+AAAAJIQH9uhq666Krp16xZ169aNpk2bxve///2YOXNmocdiO/HLX/4ycrlcDBkypNCjUI29//77ccIJJ0Tjxo2jpKQk9txzz3j++ecLPRbV1KpVq+Liiy+Otm3bRklJSeyyyy5xxRVXhHP6sCU8/fTT0a9fvygrK4tcLhf3339/peuzLItLLrkkmjdvHiUlJXHwwQfH7NmzCzNsFYiP7dBTTz0VgwcPjv/93/+Nxx57LL744os49NBDY/ny5YUejWpu2rRp8bvf/S722muvQo9CNbZ48eLo0aNH1KpVKx555JF47bXX4tprr42GDRsWejSqqVGjRsXo0aPjN7/5Tbz++usxatSo+NWvfhU33XRToUejGli+fHl85zvfiZtvvnmd1//qV7+KG2+8MW655ZZ47rnnorS0NA477LD47LPPEk9aNU61SyxcuDCaNm0aTz31VBx44IGFHodqatmyZbHPPvvEb3/72xg5cmR07tw5fv3rXxd6LKqhYcOGxdSpU+Pvf/97oUdhO3HkkUdGs2bN4tZbb80v++EPfxglJSVx++23F3AyqptcLhf33XdffP/734+If3/rUVZWFuedd14MHTo0IiLKy8ujWbNmMX78+Pjxj39cwGnXzTcfRHl5eURENGrUqMCTUJ0NHjw4+vbtGwcffHChR6Gae+CBB6Jr165x3HHHRdOmTWPvvfeOsWPHFnosqrHu3bvH448/HrNmzYqIiJdffjmeeeaZOPzwwws8GdXdnDlz4sMPP6z0b2v9+vVjv/32i2effbaAk61fzUIPQGFVVFTEkCFDokePHrHHHnsUehyqqbvuuitefPHFmDZtWqFHYTvw9ttvx+jRo+Pcc8+NCy+8MKZNmxZnn3121K5dOwYOHFjo8aiGhg0bFkuWLIkOHTpEUVFRrFq1Kq688soYMGBAoUejmvvwww8jIqJZs2aVljdr1ix/3TeN+NjODR48OF599dV45plnCj0K1dR7770X//Vf/xWPPfZY1KlTp9DjsB2oqKiIrl27xi9+8YuIiNh7773j1VdfjVtuuUV8sFXcfffdcccdd8Sdd94ZnTp1iunTp8eQIUOirKzMaw6+wm5X27EzzzwzHnzwwZgyZUq0aNGi0ONQTb3wwguxYMGC2GeffaJmzZpRs2bNeOqpp+LGG2+MmjVrxqpVqwo9ItVM8+bNY/fdd6+0rGPHjvHuu+8WaCKqu/PPPz+GDRsWP/7xj2PPPfeME088Mc4555y46qqrCj0a1dxOO+0UERHz58+vtHz+/Pn5675pxMd2KMuyOPPMM+O+++6LJ554Itq2bVvokajGvvvd78Yrr7wS06dPz/907do1BgwYENOnT4+ioqJCj0g106NHj7VOHz5r1qxo3bp1gSaiuvv000+jRo3Kb6mKioqioqKiQBOxvWjbtm3stNNO8fjjj+eXLVmyJJ577rnYf//9CzjZ+tntajs0ePDguPPOO+Mvf/lL1K1bN79PYP369aOkpKTA01Hd1K1bd63jiUpLS6Nx48aOM2KrOOecc6J79+7xi1/8Ivr37x///Oc/Y8yYMTFmzJhCj0Y11a9fv7jyyiujVatW0alTp3jppZfiuuuui1NOOaXQo1ENLFu2LN5888385Tlz5sT06dOjUaNG0apVqxgyZEiMHDkydt1112jbtm1cfPHFUVZWlj8j1jeNU+1uh3K53DqXjxs3LgYNGpR2GLZLvXv3dqpdtqoHH3wwhg8fHrNnz462bdvGueeeGz/96U8LPRbV1NKlS+Piiy+O++67LxYsWBBlZWVx/PHHxyWXXBK1a9cu9Hhs45588sno06fPWssHDhwY48ePjyzL4tJLL40xY8bEJ598EgcccED89re/jd12260A026c+AAAAJJwzAcAAJCE+AAAAJIQHwAAQBLiAwAASEJ8AAAASYgPAAAgCfEBAAAkIT4AAIAkxAdANdOmTZsq//b4XC4X999//1adZ+7cuZHL5WL69OkbvF3v3r1jyJAhW3WWqqrqzABsGvEB8A2VIgwK4cknn4xcLheffPJJoUeJiIhBgwbF97///UrLWrZsGfPmzYs99tijMEMBVFPiA4Bq6Ysvvtjs+xYVFcVOO+0UNWvW3IITASA+ADaid+/ecdZZZ8WQIUOiYcOG0axZsxg7dmwsX748Tj755Khbt260a9cuHnnkkfx9Xn311Tj88MNjhx12iGbNmsWJJ54YH330UaV1nn322XHBBRdEo0aNYqeddooRI0bkr2/Tpk1ERBxzzDGRy+Xyl9966604+uijo1mzZrHDDjtEt27dYvLkyV9r+z766KM45phj4lvf+lbsuuuu8cADD1S6fmPb8uijj8YBBxwQDRo0iMaNG8eRRx4Zb7311jofa+7cudGnT5+IiGjYsGHkcrkYNGhQ/vqKior1Picbk8vlYvTo0XHUUUdFaWlpXHnllbFq1ao49dRTo23btlFSUhLt27ePG264IX+fESNGxIQJE+Ivf/lL5HK5yOVy8eSTT66129Xqb2sef/zx6Nq1a3zrW9+K7t27x8yZMyvNMHLkyGjatGnUrVs3TjvttBg2bFh07ty5ytsAUN2JD4AqmDBhQjRp0iT++c9/xllnnRX/+Z//Gccdd1x07949XnzxxTj00EPjxBNPjE8//TQ++eSTOOigg2LvvfeO559/Ph599NGYP39+9O/ff611lpaWxnPPPRe/+tWv4vLLL4/HHnssIiKmTZsWERHjxo2LefPm5S8vW7YsjjjiiHj88cfjpZdeiu9973vRr1+/ePfddzd72y677LLo379/zJgxI4444ogYMGBAfPzxxxERVdqW5cuXx7nnnhvPP/98PP7441GjRo045phjoqKiYq3HatmyZdx7770RETFz5syYN29epRjY0HNSFSNGjIhjjjkmXnnllTjllFOioqIiWrRoEffcc0+89tprcckll8SFF14Yd999d0REDB06NPr37x/f+973Yt68eTFv3rzo3r37etd/0UUXxbXXXhvPP/981KxZM0455ZT8dXfccUdceeWVMWrUqHjhhReiVatWMXr06CrPDrBdyADYoF69emUHHHBA/vKXX36ZlZaWZieeeGJ+2bx587KIyJ599tnsiiuuyA499NBK63jvvfeyiMhmzpy5znVmWZZ169Yt++///u/85YjI7rvvvo3O16lTp+ymm27KX27dunV2/fXXV2nbIiL7+c9/nr+8bNmyLCKyRx55JMuyrErb8lULFy7MIiJ75ZVXsizLsjlz5mQRkb300ktZlmXZlClTsojIFi9eXOl+VXlONrYtQ4YM2ejtBg8enP3whz/MXx44cGB29NFHV7rN+maePHly/jYPPfRQFhHZihUrsizLsv322y8bPHhwpfX06NEj+853vlOl+QG2B775AKiCvfbaK//noqKiaNy4cey55575Zc2aNYuIiAULFsTLL78cU6ZMiR122CH/06FDh4iISrsjrbnOiIjmzZvHggULNjjHsmXLYujQodGxY8do0KBB7LDDDvH6669/rW8+1pyjtLQ06tWrl5+jKtsye/bsOP7442PnnXeOevXq5XcR25yZNuc5WVPXrl3XWnbzzTdHly5dYscdd4wddtghxowZs9nP15rzNW/ePCIiP9/MmTNj3333rXT7r14G2N45kg6gCmrVqlXpci6Xq7Qsl8tFxL+PWVi2bFn069cvRo0atdZ6Vr9hXd8617Wr0pqGDh0ajz32WFxzzTXRrl27KCkpiWOPPTZWrly5ydtUlTmqsi39+vWL1q1bx9ixY6OsrCwqKipijz322KyZNuc5WVNpaWmly3fddVcMHTo0rr322th///2jbt26cfXVV8dzzz23ybN9db41/84BqBrxAbCF7bPPPnHvvfdGmzZtvtbZkmrVqhWrVq2qtGzq1KkxaNCgOOaYYyLi33Ewd+7crzPuBm1sWxYtWhQzZ86MsWPHRs+ePSMi4plnntngOmvXrh0Rsda2bQ1Tp06N7t27xxlnnJFf9tWD4WvXrr1FZmnfvn1MmzYtTjrppPyy1cfqAPBvdrsC2MIGDx4cH3/8cRx//PExbdq0eOutt2LSpElx8sknb9Kb3DZt2sTjjz8eH374YSxevDgiInbdddf485//HNOnT4+XX345fvKTn2zVT943ti0NGzaMxo0bx5gxY+LNN9+MJ554Is4999wNrrN169aRy+XiwQcfjIULF8ayZcu22vy77rprPP/88zFp0qSYNWtWXHzxxWsFQZs2bWLGjBkxc+bM+Oijjzb7FL1nnXVW3HrrrTFhwoSYPXt2jBw5MmbMmJH/hgQA8QGwxZWVlcXUqVNj1apVceihh8aee+4ZQ4YMiQYNGkSNGlX/v91rr702HnvssWjZsmXsvffeERFx3XXXRcOGDaN79+7Rr1+/OOyww2KfffbZWpuy0W2pUaNG3HXXXfHCCy/EHnvsEeecc05cffXVG1znt7/97bjsssti2LBh0axZszjzzDO32vw/+9nP4gc/+EH86Ec/iv322y8WLVpU6VuQiIif/vSn0b59++jatWvsuOOOMXXq1M16rAEDBsTw4cNj6NChsc8++8ScOXNi0KBBUadOnS2xKQDVQi7LsqzQQwBAdXTIIYfETjvtFBMnTiz0KADfCI75AIAt4NNPP41bbrklDjvssCgqKoo//OEPMXny5E36PSUA1Z3drgCqqTvuuKPSKXLX/OnUqVOhx9sk28K25HK5ePjhh+PAAw+MLl26xF//+te499574+CDDy70aADfGHa7Aqimli5dGvPnz1/ndbVq1YrWrVsnnmjzVadtAdieiQ8AACAJu10BAABJiA8AACAJ8QEAACQhPgAAgCTEBwAAkIT4AAAAkhAfAABAEv8PXqtRKcOqR1sAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAx8AAAIjCAYAAABia6bHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAMTBJREFUeJzt3X+81/P9+P/76VSnM/oh/dbvohIR9SZGM5kSY2PSO1OqYertx/wYa/ohls0wNmxtb+o9vybGTFrJr4b8lvyaQn5TSL8U0Xl8//Dt9XEUTuk8jpPr9XI5F53n8/l6nsfrcZ7qdTvP1/N5ilJKKQAAACpZjaoeAAAA8M0gPgAAgCzEBwAAkIX4AAAAshAfAABAFuIDAADIQnwAAABZiA8AACAL8QEAAGQhPgCq0EsvvRRFRUUxadKkqh5KOf/6179i5513jjp16kRRUVEsWbKkqocEwGZAfACbhUmTJkVRUVG5jyZNmsQ+++wT06ZNyz6eu+++u9xYatWqFe3bt4+jjjoqXnzxxU3yNe6///4YO3bsJg+Dd999Nw4//PAoLS2NSy+9NP7617/GFltssUm/BgDfTDWregAAm9LZZ58d7dq1i5RSLFy4MCZNmhQHHHBA/POf/4wDDzww+3hOOOGE6NmzZ3z00Ufx2GOPxcSJE2Pq1Knx5JNPRosWLb7Svu+///4YN25cDBkyJBo0aLBpBhwRDz/8cCxfvjzGjx8fffr02WT7BQDxAWxW+vXrFz169Ch8PmzYsGjatGlce+21VRIfe+21Vxx22GEREXH00UfHdtttFyeccEJMnjw5zjzzzOzjqYhFixZFRGzSoGHDpZTigw8+iNLS0qoeCsAm421XwGatQYMGUVpaGjVrlv9Zy/vvvx+nnHJKtGrVKkpKSqJTp07x29/+NlJKERGxatWq6Ny5c3Tu3DlWrVpVeNzixYujefPmsccee8SaNWs2eDzf/e53IyJiwYIFX7jdnXfeGXvttVdsscUW0aBBgzj44IPj2WefLawfO3ZsnHbaaRER0a5du8Lbu1566aUv3O+UKVNi1113jdLS0mjUqFEceeSR8frrrxfWf+c734nBgwdHRETPnj2jqKgohgwZ8oX7fP3112Po0KHRtGnTKCkpia5du8YVV1xRWL8hczl37twYMmRItG/fPurUqRPNmjWLoUOHxrvvvlvua44dOzaKiopi3rx5ceSRR0b9+vWjcePGcdZZZ0VKKV599dU4+OCDo169etGsWbO44IILvvA5rM8jjzwS+++/fzRq1ChKS0ujXbt2MXTo0HLblJWVxcUXXxw77rhj1KlTJxo3bhx9+/aNRx55pLDNxx9/HOPHj48OHTpESUlJtG3bNn7xi1/Ehx9+WG5fbdu2jQMPPDCmT58ePXr0iNLS0vjTn/4UERFLliyJk046qXC8duzYMX79619HWVnZBj8vgKrkzAewWVm6dGm88847kVKKRYsWxe9///tYsWJFHHnkkYVtUkrx/e9/P+66664YNmxY7LzzzjF9+vQ47bTT4vXXX4+LLrooSktLY/LkybHnnnvGqFGj4sILL4yIiBEjRsTSpUtj0qRJUVxcvMHje+GFFyIiYuutt/7cbWbOnBn9+vWL9u3bx9ixY2PVqlXx+9//Pvbcc8947LHHom3btvHDH/4w5s2bF9dee21cdNFF0ahRo4iIaNy48efud9KkSXH00UdHz549Y8KECbFw4cK4+OKL47777ovHH388GjRoEKNGjYpOnTrFxIkTC29h69Chw+fuc+HChbH77rtHUVFRjBw5Mho3bhzTpk2LYcOGxbJly+Kkk07aoLm8/fbb48UXX4yjjz46mjVrFk8//XRMnDgxnn766XjggQeiqKio3NcfMGBAdOnSJc4777yYOnVqnHPOOdGwYcP405/+FN/97nfj17/+dVx99dVx6qmnRs+ePWPvvfeu0Pdp0aJF8b3vfS8aN24cZ5xxRjRo0CBeeuml+Pvf/15uu2HDhsWkSZOiX79+MXz48Pj444/j3//+dzzwwAOFM3DDhw+PyZMnx2GHHRannHJKPPjggzFhwoR49tln46abbiq3v+eeey4GDhwYxx57bPzkJz+JTp06xcqVK6N3797x+uuvx7HHHhutW7eO+++/P84888x4880343e/+12FnhPA10IC2AxceeWVKSLW+SgpKUmTJk0qt+3NN9+cIiKdc8455ZYfdthhqaioKD3//POFZWeeeWaqUaNGmjVrVpoyZUqKiPS73/3uS8dz1113pYhIV1xxRXr77bfTG2+8kaZOnZratm2bioqK0sMPP5xSSmnBggUpItKVV15ZeOzOO++cmjRpkt59993CsieeeCLVqFEjHXXUUYVl559/foqItGDBgi8dz+rVq1OTJk3SDjvskFatWlVYfuutt6aISKNHjy4sWzuXa8f4RYYNG5aaN2+e3nnnnXLLjzjiiFS/fv20cuXKwrKKzOWnt1/r2muvTRGRZs2aVVg2ZsyYFBHpmGOOKSz7+OOPU8uWLVNRUVE677zzCsvfe++9VFpamgYPHvylz2etm2666Uvn4M4770wRkU444YR11pWVlaWUUpozZ06KiDR8+PBy60899dQUEenOO+8sLGvTpk2KiPSvf/2r3Lbjx49PW2yxRZo3b1655WeccUYqLi5Or7zySoWfF0BV87YrYLNy6aWXxu233x633357XHXVVbHPPvvE8OHDy/3E+rbbbovi4uI44YQTyj32lFNOiZRSubtjjR07Nrp27RqDBw+O448/Pnr37r3O477I0KFDo3HjxtGiRYvo379/vP/++zF58uRy16V82ptvvhlz5syJIUOGRMOGDQvLu3XrFvvtt1/cdtttFf7an/bII4/EokWL4vjjj486deoUlvfv3z86d+4cU6dO3eB9ppTixhtvjIMOOihSSvHOO+8UPvbff/9YunRpPPbYY4XtKzKXn76+4YMPPoh33nkndt9994iIcvtaa/jw4YU/FxcXR48ePSKlFMOGDSssb9CgQXTq1GmD7jK29nqXW2+9NT766KP1bnPjjTdGUVFRjBkzZp11a8/QrP1+/exnPyu3/pRTTomIWGfe27VrF/vvv3+5ZVOmTIm99torttpqq3Jz3KdPn1izZk3MmjWrws8LoKp52xWwWfmv//qvci/sBw4cGN27d4+RI0fGgQceGLVr146XX345WrRoEXXr1i332C5dukRExMsvv1xYVrt27bjiiiuiZ8+eUadOnbjyyivXeevPFxk9enTstddeUVxcHI0aNYouXbqsc/3Jp6392p06dVpnXZcuXWL69Onx/vvvb/Ctb79ov507d4577713g/YXEfH222/HkiVLYuLEiTFx4sT1brP24vWIis3l4sWLY9y4cXHdddeVe2zEJ2+p+6zWrVuX+7x+/fpRp06dwtvQPr38s9eNfJHevXvHoYceGuPGjYuLLroovvOd78QhhxwS//3f/x0lJSUR8clb6Fq0aFEuEj/r5Zdfjho1akTHjh3LLW/WrFk0aNCg3LEW8Ul8fNb8+fNj7ty5n/uWus/OE8DXmfgANms1atSIffbZJy6++OKYP39+dO3adYP3MX369Ij45Cfx8+fPX+8LxM+z4447bra3q117sfORRx5ZuEj9s7p161bu8y+by8MPPzzuv//+OO2002LnnXeOLbfcMsrKyqJv377rvbh6fdfdfN61OOn/v5lARRQVFcUNN9wQDzzwQPzzn/+M6dOnx9ChQ+OCCy6IBx54ILbccssK72vt/ipifXe2Kisri/322y9OP/309T5mu+2226CxAFQl8QFs9j7++OOIiFixYkVERLRp0yZmzpwZy5cvL3f24z//+U9h/Vpz586Ns88+O44++uiYM2dODB8+PJ588smoX79+pYx17dd+7rnn1ln3n//8Jxo1alQ467EhZ2A+vd+1d9xa67nnniv3nCuqcePGUbdu3VizZk2FAuvL5vK9996LO+64I8aNGxejR48uPG7+/PkbPLZNZffdd4/dd989zj333Ljmmmti0KBBcd1118Xw4cOjQ4cOMX369Fi8ePHnnv1o06ZNlJWVxfz58wtn1iI+uVB/yZIlFZr3Dh06xIoVKzbbiAW+WVzzAWzWPvroo5gxY0bUrl278OLvgAMOiDVr1sQf/vCHcttedNFFUVRUFP369Ss8dsiQIdGiRYu4+OKLY9KkSbFw4cI4+eSTK228zZs3j5133jkmT55c7jeXP/XUUzFjxow44IADCsvWRkhFfsN5jx49okmTJvHHP/6x3C1ep02bFs8++2z0799/g8daXFwchx56aNx4443x1FNPrbP+7bffLvy5InO59ozFZ89QVMXdnN577711xrHzzjtHRBTm79BDD42UUowbN26dx6997Nrv12efw9o7flVk3g8//PCYPXt24azRpy1ZsqQQ1wDVgTMfwGZl2rRphTMYixYtimuuuSbmz58fZ5xxRtSrVy8iIg466KDYZ599YtSoUfHSSy/FTjvtFDNmzIh//OMfcdJJJxVuLXvOOefEnDlz4o477oi6detGt27dYvTo0fHLX/4yDjvssHIhsCmdf/750a9fv+jVq1cMGzascKvd+vXrx9ixYwvb7brrrhERMWrUqDjiiCOiVq1acdBBB633epBatWrFr3/96zj66KOjd+/eMXDgwMKtdtu2bbvRQXXeeefFXXfdFbvttlv85Cc/ie233z4WL14cjz32WMycOTMWL14cERWby3r16sXee+8dv/nNb+Kjjz6KbbbZJmbMmPGlvxOlMkye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t27daNKkSRxyyCHx3HPPldvmgw8+iBEjRsTWW28dW265ZRx66KGxcOHCKhpx9XH55ZdHt27dCr/YplevXjFt2rTCevO6aZx33nlRVFQUJ510UmGZud1wY8eOjaKionIfnTt3Lqw3pxvv9ddfjyOPPDK23nrrKC0tjR133DEeeeSRwvqUUowePTqaN28epaWl0adPn5g/f34Vjrh6aNu27TrHbFFRUYwYMSIiHLMba82aNXHWWWdFu3btorS0NDp06BDjx4+PT99/xzG7cZYvXx4nnXRStGnTJkpLS2OPPfaIhx9+uLDevFbMrFmz4qCDDooWLVpEUVFR3HzzzeXWV2QeFy9eHIMGDYp69epFgwYNYtiwYbFixYqMz+ILpM3c/vvvn6688sr01FNPpTlz5qQDDjggtW7dOq1YsaKwzXHHHZdatWqV7rjjjvTII4+k3XffPe2xxx5VOOrq4ZZbbklTp05N8+bNS88991z6xS9+kWrVqpWeeuqplJJ53RQeeuih1LZt29StW7d04oknFpab2w03ZsyY1LVr1/Tmm28WPt5+++3CenO6cRYvXpzatGmThgwZkh588MH04osvpunTp6fnn3++sM15552X6tevn26++eb0xBNPpO9///upXbt2adWqVVU48q+/RYsWlTteb7/99hQR6a677kopOWY31rnnnpu23nrrdOutt6YFCxakKVOmpC233DJdfPHFhW0csxvn8MMPT9tvv32655570vz589OYMWNSvXr10muvvZZSMq8Vddttt6VRo0alv//97yki0k033VRufUXmsW/fvmmnnXZKDzzwQPr3v/+dOnbsmAYOHJj5mazfZh8fn7Vo0aIUEemee+5JKaW0ZMmSVKtWrTRlypTCNs8++2yKiDR79uyqGma1tdVWW6W//OUv5nUTWL58edp2223T7bffnnr37l2ID3O7ccaMGZN22mmn9a4zpxvv5z//efr2t7/9uevLyspSs2bN0vnnn19YtmTJklRSUpKuvfbaHEPcbJx44ompQ4cOqayszDH7FfTv3z8NHTq03LIf/vCHadCgQSklx+zGWrlyZSouLk633nprueW77LJLGjVqlHndSJ+Nj4rM4zPPPJMiIj388MOFbaZNm5aKiorS66+/nm3sn2ezf9vVZy1dujQiIho2bBgREY8++mh89NFH0adPn8I2nTt3jtatW8fs2bOrZIzV0Zo1a+K6666L999/P3r16mVeN4ERI0ZE//79y81hhGP2q5g/f360aNEi2rdvH4MGDYpXXnklIszpV3HLLbdEjx494kc/+lE0adIkunfvHn/+858L6xcsWBBvvfVWubmtX79+7LbbbuZ2A6xevTquuuqqGDp0aBQVFTlmv4I99tgj7rjjjpg3b15ERDzxxBNx7733Rr9+/SLCMbuxPv7441izZk3UqVOn3PLS0tK49957zesmUpF5nD17djRo0CB69OhR2KZPnz5Ro0aNePDBB7OP+bNqVvUAciorK4uTTjop9txzz9hhhx0iIuKtt96K2rVrR4MGDcpt27Rp03jrrbeqYJTVy5NPPhm9evWKDz74ILbccsu46aabYvvtt485c+aY16/guuuui8cee6zce2XXcsxunN122y0mTZoUnTp1ijfffDPGjRsXe+21Vzz11FPm9Ct48cUX4/LLL4+f/exn8Ytf/CIefvjhOOGEE6J27doxePDgwvw1bdq03OPM7Ya5+eabY8mSJTFkyJCI8PfAV3HGGWfEsmXLonPnzlFcXBxr1qyJc889NwYNGhQR4ZjdSHXr1o1evXrF+PHjo0uXLtG0adO49tprY/bs2dGxY0fzuolUZB7feuutaNKkSbn1NWvWjIYNG34t5vobFR8jRoyIp556Ku69996qHspmo1OnTjFnzpxYunRp3HDDDTF48OC45557qnpY1dqrr74aJ554Ytx+++3r/ASJjbf2p5oREd26dYvddtst2rRpE9dff32UlpZW4ciqt7KysujRo0f86le/ioiI7t27x1NPPRV//OMfY/DgwVU8us3H//7v/0a/fv2iRYsWVT2Uau/666+Pq6++Oq655pro2rVrzJkzJ0466aRo0aKFY/Yr+utf/xpDhw6NbbbZJoqLi2OXXXaJgQMHxqOPPlrVQ+Nr5BvztquRI0fGrbfeGnfddVe0bNmysLxZs2axevXqWLJkSbntFy5cGM2aNcs8yuqndu3a0bFjx9h1111jwoQJsdNOO8XFF19sXr+CRx99NBYtWhS77LJL1KxZM2rWrBn33HNPXHLJJVGzZs1o2rSpud0EGjRoENttt108//zzjtevoHnz5rH99tuXW9alS5fCW9rWzt9n78Jkbivu5ZdfjpkzZ8bw4cMLyxyzG++0006LM844I4444ojYcccd48c//nGcfPLJMWHChIhwzH4VHTp0iHvuuSdWrFgRr776ajz00EPx0UcfRfv27c3rJlKReWzWrFksWrSo3PqPP/44Fi9e/LWY680+PlJKMXLkyLjpppvizjvvjHbt2pVbv+uuu0atWrXijjvuKCx77rnn4pVXXolevXrlHm61V1ZWFh9++KF5/Qr23XffePLJJ2POnDmFjx49esSgQYMKfza3X92KFSvihRdeiObNmztev4I999xznduXz5s3L9q0aRMREe3atYtmzZqVm9tly5bFgw8+aG4r6Morr4wmTZpE//79C8scsxtv5cqVUaNG+Zc/xcXFUVZWFhGO2U1hiy22iObNm8d7770X06dPj4MPPti8biIVmcdevXrFkiVLyp1xuvPOO6OsrCx222237GNeR1Vf8V7ZfvrTn6b69eunu+++u9wtC1euXFnY5rjjjkutW7dOd955Z3rkkUdSr169Uq9evapw1NXDGWecke655560YMGCNHfu3HTGGWekoqKiNGPGjJSSed2UPn23q5TM7cY45ZRT0t13350WLFiQ7rvvvtSnT5/UqFGjtGjRopSSOd1YDz30UKpZs2Y699xz0/z589PVV1+dvvWtb6WrrrqqsM15552XGjRokP7xj3+kuXPnpoMPPtjtNStozZo1qXXr1unnP//5Ouscsxtn8ODBaZtttincavfvf/97atSoUTr99NML2zhmN86//vWvNG3atPTiiy+mGTNmpJ122inttttuafXq1Skl81pRy5cvT48//nh6/PHHU0SkCy+8MD3++OPp5ZdfTilVbB779u2bunfvnh588MF07733pm233datdnOJiPV+XHnllYVtVq1alY4//vi01VZbpW9961vpBz/4QXrzzTerbtDVxNChQ1ObNm1S7dq1U+PGjdO+++5bCI+UzOum9Nn4MLcbbsCAAal58+apdu3aaZtttkkDBgwo97sozOnG++c//5l22GGHVFJSkjp37pwmTpxYbn1ZWVk666yzUtOmTVNJSUnad99903PPPVdFo61epk+fniJivfPlmN04y5YtSyeeeGJq3bp1qlOnTmrfvn0aNWpU+vDDDwvbOGY3zt/+9rfUvn37VLt27dSsWbM0YsSItGTJksJ681oxd91113pfuw4ePDilVLF5fPfdd9PAgQPTlltumerVq5eOPvrotHz58ip4NusqSulTv9ITAACgkmz213wAAABfD+IDAADIQnwAAABZiA8AACAL8QEAAGQhPgAAgCzEBwAAkIX4AAAAshAfAABAFuIDAADIQnwAUO2tXr26qocAQAWID4BqoKysLCZMmBDt2rWL0tLS2GmnneKGG26IlFL06dMn9t9//0gpRUTE4sWLo2XLljF69OiIiFizZk0MGzas8NhOnTrFxRdfXG7/Q4YMiUMOOSR+9atfRdOmTaNBgwZx9tlnx8cffxynnXZaNGzYMFq2bBlXXnllhca7evXqGDlyZDRv3jzq1KkTbdq0iQkTJhTWL1myJI499tho2rRp1KlTJ3bYYYe49dZbC+tvvPHG6Nq1a5SUlETbtm3jggsuKLf/tm3bxvjx4+Ooo46KevXqxTHHHBMREffee2/stddeUVpaGq1atYoTTjgh3n///Q2fcAAqRc2qHgAAX27ChAlx1VVXxR//+MfYdtttY9asWXHkkUdG48aNY/LkybHjjjvGJZdcEieeeGIcd9xxsc022xTio6ysLFq2bBlTpkyJrbfeOu6///445phjonnz5nH44YcXvsadd94ZLVu2jFmzZsV9990Xw4YNi/vvvz/23nvvePDBB+Nvf/tbHHvssbHffvtFy5Ytv3C8l1xySdxyyy1x/fXXR+vWrePVV1+NV199tTCefv36xfLly+Oqq66KDh06xDPPPBPFxcUREfHoo4/G4YcfHmPHjo0BAwbE/fffH8cff3xsvfXWMWTIkMLX+O1vfxujR4+OMWPGRETECy+8EH379o1zzjknrrjiinj77bdj5MiRMXLkyApHEwCVqyit/VEZAF9LH374YTRs2DBmzpwZvXr1KiwfPnx4rFy5Mq655pqYMmVKHHXUUXHSSSfF73//+3j88cdj2223/dx9jhw5Mt5666244YYbIuKTMx933313vPjii1GjxicnxTt37hxNmjSJWbNmRcQnZ1Dq168ff/nLX+KII474wjGfcMIJ8fTTT8fMmTOjqKio3LoZM2ZEv3794tlnn43ttttunccOGjQo3n777ZgxY0Zh2emnnx5Tp06Np59+OiI+OfPRvXv3uOmmm8rNR3FxcfzpT38qLLv33nujd+/e8f7770edOnW+cMwAVD5nPgC+5p5//vlYuXJl7LfffuWWr169Orp37x4RET/60Y/ipptuivPOOy8uv/zydcLj0ksvjSuuuCJeeeWVWLVqVaxevTp23nnnctt07dq1EB4REU2bNo0ddtih8HlxcXFsvfXWsWjRoi8d85AhQ2K//faLTp06Rd++fePAAw+M733vexERMWfOnGjZsuV6wyMi4tlnn42DDz643LI999wzfve738WaNWsKZ0h69OhRbpsnnngi5s6dG1dffXVhWUopysrKYsGCBdGlS5cvHTcAlUt8AHzNrVixIiIipk6dGttss025dSUlJRERsXLlynj00UejuLg45s+fX26b6667Lk499dS44IILolevXlG3bt04//zz48EHHyy3Xa1atcp9XlRUtN5lZWVlXzrmXXbZJRYsWBDTpk2LmTNnxuGHHx59+vSJG264IUpLSyv2xL/EFltsUe7zFStWxLHHHhsnnHDCOtu2bt16k3xNAL4a8QHwNbf99ttHSUlJvPLKK9G7d+/1bnPKKadEjRo1Ytq0aXHAAQdE//7947vf/W5ERNx3332xxx57xPHHH1/Y/oUXXqj0cderVy8GDBgQAwYMiMMOOyz69u0bixcvjm7dusVrr70W8+bNW+/Zjy5dusR9991Xbtl9990X2223XeGsx/rssssu8cwzz0THjh03+XMBYNMQHwBfc3Xr1o1TTz01Tj755CgrK4tvf/vbsXTp0rjvvvuiXr160ahRo7jiiiti9uzZscsuu8Rpp50WgwcPjrlz58ZWW20V2267bfzf//1fTJ8+Pdq1axd//etf4+GHH4527dpV2pgvvPDCaN68eXTv3j1q1KgRU6ZMiWbNmkWDBg2id+/esffee8ehhx4aF154YXTs2DH+85//RFFRUfTt2zdOOeWU6NmzZ4wfPz4GDBgQs2fPjj/84Q9x2WWXfeHX/PnPfx677757jBw5MoYPHx5bbLFFPPPMM3H77bfHH/7wh0p7rgBUnFvtAlQD48ePj7POOismTJgQXbp0ib59+8bUqVOjbdu2MWzYsBg7dmzssssuERExbty4aNq0aRx33HEREXHsscfGD3/4wxgwYEDstttu8e6775Y7C1IZ6tatG7/5zW+iR48e0bNnz3jppZfitttuK1xTcuONN0bPnj1j4MCBsf3228fpp58ea9asiYhPzmBcf/31cd1118UOO+wQo0ePjrPPPrvcna7Wp1u3bnHPPffEvHnzYq+99oru3bvH6NGjo0WLFpX6XAGoOHe7AgAAsnDmAwAAyEJ8ALDBfvWrX8WWW2653o9+/fpV9fAA+JrytisANtjixYtj8eLF611XWlq6zi2BASBCfAAAAJl42xUAAJCF+AAAALIQHwAAQBbiAwAAyEJ8AAAAWYgPAAAgC/EBAABk8f8BU4j5pFkDUX8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Q1 = df[numeric_features].quantile(0.25)\n",
        "Q3 = df[numeric_features].quantile(0.75)\n",
        "IQR = Q3 - Q1"
      ],
      "metadata": {
        "id": "e6RJJVHvKkuq"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik\n",
        "condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)\n",
        "df_filtered_numeric = df.loc[condition, numeric_features]\n",
        "\n",
        "# Menggabungkan kembali dengan kolom kategorikal\n",
        "categorical_features = df.select_dtypes(include=['object']).columns\n",
        "df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)"
      ],
      "metadata": {
        "id": "JtB5vf9gKni8"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Data Transformation**"
      ],
      "metadata": {
        "id": "3AR-bVkCBJpZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "encoder = LabelEncoder()\n",
        "\n",
        "df['diet_quality_encoded'] = encoder.fit_transform(df['diet_quality'])\n",
        "\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "4BsHwpHWBQoB",
        "outputId": "e4404f3d-24f4-474d-dfb9-1095b487a03f"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age  study_hours_per_day  attendance_percentage  sleep_hours  \\\n",
              "0   23                  0.0                   85.0          8.0   \n",
              "1   20                  6.9                   97.3          4.6   \n",
              "2   21                  1.4                   94.8          8.0   \n",
              "3   23                  1.0                   71.0          9.2   \n",
              "4   19                  5.0                   90.9          4.9   \n",
              "\n",
              "   exercise_frequency  mental_health_rating  exam_score diet_quality  \\\n",
              "0                   6                     8        56.2         Fair   \n",
              "1                   6                     8       100.0         Good   \n",
              "2                   1                     1        34.3         Poor   \n",
              "3                   4                     1        26.8         Poor   \n",
              "4                   3                     1        66.4         Fair   \n",
              "\n",
              "   diet_quality_encoded  \n",
              "0                     0  \n",
              "1                     1  \n",
              "2                     2  \n",
              "3                     2  \n",
              "4                     0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a29c5beb-558d-4525-823f-04405aa72041\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>study_hours_per_day</th>\n",
              "      <th>attendance_percentage</th>\n",
              "      <th>sleep_hours</th>\n",
              "      <th>exercise_frequency</th>\n",
              "      <th>mental_health_rating</th>\n",
              "      <th>exam_score</th>\n",
              "      <th>diet_quality</th>\n",
              "      <th>diet_quality_encoded</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>23</td>\n",
              "      <td>0.0</td>\n",
              "      <td>85.0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>56.2</td>\n",
              "      <td>Fair</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20</td>\n",
              "      <td>6.9</td>\n",
              "      <td>97.3</td>\n",
              "      <td>4.6</td>\n",
              "      <td>6</td>\n",
              "      <td>8</td>\n",
              "      <td>100.0</td>\n",
              "      <td>Good</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>21</td>\n",
              "      <td>1.4</td>\n",
              "      <td>94.8</td>\n",
              "      <td>8.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>34.3</td>\n",
              "      <td>Poor</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>23</td>\n",
              "      <td>1.0</td>\n",
              "      <td>71.0</td>\n",
              "      <td>9.2</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>26.8</td>\n",
              "      <td>Poor</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>19</td>\n",
              "      <td>5.0</td>\n",
              "      <td>90.9</td>\n",
              "      <td>4.9</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>66.4</td>\n",
              "      <td>Fair</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a29c5beb-558d-4525-823f-04405aa72041')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-a29c5beb-558d-4525-823f-04405aa72041 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-a29c5beb-558d-4525-823f-04405aa72041');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-2b8b6aca-6f5c-4494-b217-dafb8a819c4d\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-2b8b6aca-6f5c-4494-b217-dafb8a819c4d')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-2b8b6aca-6f5c-4494-b217-dafb8a819c4d button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 986,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 17,\n        \"max\": 24,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          20,\n          18,\n          23\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"study_hours_per_day\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.4302980582721059,\n        \"min\": 0.0,\n        \"max\": 7.3,\n        \"num_unique_values\": 72,\n        \"samples\": [\n          5.0,\n          6.5,\n          3.8\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"attendance_percentage\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.314611972074811,\n        \"min\": 59.5,\n        \"max\": 100.0,\n        \"num_unique_values\": 317,\n        \"samples\": [\n          72.7,\n          96.3,\n          74.7\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sleep_hours\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.2206089605999642,\n        \"min\": 3.2,\n        \"max\": 9.8,\n        \"num_unique_values\": 67,\n        \"samples\": [\n          6.2,\n          6.0,\n          7.4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exercise_frequency\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          6,\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"mental_health_rating\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 1,\n        \"max\": 10,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          2,\n          1,\n          9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exam_score\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 16.649865256078876,\n        \"min\": 26.2,\n        \"max\": 100.0,\n        \"num_unique_values\": 476,\n        \"samples\": [\n          47.5,\n          58.1,\n          65.2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"diet_quality\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Fair\",\n          \"Good\",\n          \"Poor\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"diet_quality_encoded\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0,\n          1,\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(columns=['diet_quality'], inplace=True)"
      ],
      "metadata": {
        "id": "dE1FpJ6zm1xv"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Exploratory Data Analysis**"
      ],
      "metadata": {
        "id": "0XAAAZ605AVG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mental_health = df.groupby('mental_health_rating')[['exam_score']].mean()\n",
        "print(mental_health)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GBm_YMRR5Fb3",
        "outputId": "bf37dd47-ebe7-4770-d6b3-88490e9ca906"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                      exam_score\n",
            "mental_health_rating            \n",
            "1                      62.759406\n",
            "2                      63.207527\n",
            "3                      63.785437\n",
            "4                      65.957407\n",
            "5                      66.430612\n",
            "6                      70.882075\n",
            "7                      73.994382\n",
            "8                      74.447115\n",
            "9                      76.620930\n",
            "10                     77.727551\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mental_health.plot(kind='bar', figsize=(8, 5))\n",
        "plt.title('Distribusi Antar Mental Health dan Nilai Siswa')\n",
        "plt.xlabel('Rating')\n",
        "plt.ylabel('Nilai Siswa')\n",
        "plt.xticks(rotation=0)\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "tz1g5MqP6pz9",
        "outputId": "2d1773ba-2496-4390-dcc5-b56f2b7b5e6a"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAq4AAAHWCAYAAAC2Zgs3AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAUB9JREFUeJzt3Xt8z/X///H7e7OTzTaTnRjGVs5ySnP+sCyhRE6phkqfTyhEtQrlNFRO5VDSKpFQCpVDcihGCBUltGzFJmWbQ9vYXr8/+u7987ZhY9trL27Xy+V1ufR+vl7v1+vxfO/d3Pd8P9/Pl80wDEMAAABAKedkdgEAAABAQRBcAQAAYAkEVwAAAFgCwRUAAACWQHAFAACAJRBcAQAAYAkEVwAAAFgCwRUAAACWQHAFAACAJRBccUN68cUXZbPZSuRabdu2Vdu2be2PN27cKJvNpmXLlpXI9XO98847stls+u2330r0uigav/32m2w2m9555x2zS7miwrzXco/duXNnkdbQr18/VatWrUjPWVzyq9Vms+nFF18s9Llyf79s3LixSGqTpGrVqqlfv35Fdj7gWhBcYXm5//Dlbu7u7goODlZUVJRmzpypU6dOFcl1jh49qhdffFF79uwpkvNZwdNPPy2bzaZevXpd87kWLVqk6dOnX3tRhVCtWjXZbDZFRkbmu3/evHn2901RB6eLzZ49u0RC55X+MOrXr5+8vLyKvY6LlVT/zZb7B4bNZtNHH32UZ3/uH80nTpwwoTpHP/zwg+677z5VrVpV7u7uqlSpku644w699tprZpcGXBLBFdeNsWPHasGCBZozZ46GDBkiSRo6dKjq1aun77//3uHYF154Qf/880+hzn/06FG99NJLhQ6ua9eu1dq1awv1nOLw4IMP6p9//lHVqlULdLxhGPrggw9UrVo1rVy58pr/ADAjuEqSu7u7NmzYoOTk5Dz7Fi5cKHd39xKp40YJbpdyI/Z/7NixMgzjisfNmzdPBw4cKJJrtm7dWv/8849at2592eO2bt2qJk2aaO/evXr00Uf1+uuv65FHHpGTk5NmzJjhcOyBAwc0b968IqkPuFZlzC4AKCodO3ZUkyZN7I9jYmL01VdfqXPnzrr77rv1008/ycPDQ5JUpkwZlSlTvG//s2fPqmzZsnJ1dS3W6xSUs7OznJ2dC3z8xo0b9fvvv+urr75SVFSUPv74Y0VHRxdjhYV3/vx55eTkXPY1btGihXbs2KEPP/xQTz75pL39999/19dff617770335Ex4Frceuut2rNnj5YvX65u3bpd9lgXF5ciu66Tk1OB/hibMGGCfHx8tGPHDvn6+jrsO378uMNjNze3IqsPuFaMuOK61q5dO40aNUpHjhzR+++/b2/Pb47runXr1LJlS/n6+srLy0u33HKLnnvuOUn/hrimTZtKkvr372//KDB3BKlt27aqW7eudu3apdatW6ts2bL25148xzVXdna2nnvuOQUGBsrT01N33323kpKSHI651Nyy/M752muvqU6dOipbtqzKly+vJk2aaNGiRfb9hZ3junDhQtWuXVv/+c9/FBkZqYULF+Y5Jvdj6SVLlmjChAmqXLmy3N3d1b59ex06dMih3s8++0xHjhyxv3a5c/qysrI0evRoNW7cWD4+PvL09FSrVq20YcMGh2vlfgT7yiuvaPr06apRo4bc3Ny0f//+y/bD3d1d3bp1c3gtJOmDDz5Q+fLlFRUVle/zfv75Z913333y8/OTu7u7mjRpohUrVjgck/uabtmyRcOHD1fFihXl6empe++9V3/++af9uGrVqmnfvn3atGmTvf+5P7+///5bI0aMUL169eTl5SVvb2917NhRe/fuvWy/itoXX3yhVq1aydPTU+XKlVOnTp20b98+h2O+//579evXT9WrV5e7u7sCAwM1YMAA/fXXX5c99+X6nyszM/Oyr+HlfPLJJ6pbt67c3d1Vt25dLV++PN/jXnnlFTVv3lwVKlSQh4eHGjdunO+UCpvNpsGDB9vP6+bmpjp16mj16tUFqkeSevfurZtvvrlAo64FmY975MgRPf7447rlllvk4eGhChUqqEePHnn+fy7oHNfDhw+rTp06eUKrJPn7+zs8vvj30Llz5/TSSy8pPDxc7u7uqlChglq2bKl169ZJklasWCGbzebwSddHH30km82WJ8TXqlXLYSpSXFyc2rVrJ39/f7m5ual27dqaM2fOZfuCGwsjrrjuPfjgg3ruuee0du1aPfroo/kes2/fPnXu3Fn169fX2LFj5ebmpkOHDmnLli2S/v3lOnbsWI0ePVoDBw5Uq1atJEnNmze3n+Ovv/5Sx44d1bt3bz3wwAMKCAi4bF0TJkyQzWbTM888o+PHj2v69OmKjIzUnj177CPDBTVv3jw98cQTuu+++/Tkk08qIyND33//vbZv367777+/UOeS/g0RH330kZ566ilJUp8+fdS/f38lJycrMDAwz/GTJk2Sk5OTRowYobS0NE2ZMkV9+/bV9u3bJUnPP/+80tLS9Pvvv2vatGmSZJ9nmZ6errfeekt9+vTRo48+qlOnTmn+/PmKiorSt99+q1tvvdXhWnFxccrIyNDAgQPl5uYmPz+/K/bn/vvvV4cOHXT48GHVqFFD0r9TF+677758R7v27dunFi1aqFKlSnr22Wfl6empJUuWqGvXrvroo4907733Ohw/ZMgQlS9fXmPGjNFvv/2m6dOna/Dgwfrwww8lSdOnT9eQIUPk5eWl559/XpLs749ff/1Vn3zyiXr06KHQ0FClpKTojTfeUJs2bbR//34FBwdfsX/5OXXqVL7zKDMzM/O0LViwQNHR0YqKitLkyZN19uxZzZkzRy1bttTu3bvtoWrdunX69ddf1b9/fwUGBmrfvn168803tW/fPm3btu2SX3i8XP8L+hpeytq1a9W9e3fVrl1bsbGx+uuvv9S/f39Vrlw5z7EzZszQ3Xffrb59+yorK0uLFy9Wjx49tGrVKnXq1Mnh2G+++UYff/yxHn/8cZUrV04zZ85U9+7dlZiYqAoVKly2JunfTzheeOEFPfTQQwUadb2SHTt2aOvWrerdu7cqV66s3377TXPmzFHbtm21f/9+lS1btlDnq1q1quLj4/Xjjz+qbt26hXruiy++qNjYWD3yyCO67bbblJ6erp07d+q7777THXfcoZYtW8pms2nz5s2qX7++JOnrr7+Wk5OTvvnmG/t5/vzzT/38888aPHiwvW3OnDmqU6eO7r77bpUpU0YrV67U448/rpycHA0aNKhQdeI6ZQAWFxcXZ0gyduzYccljfHx8jIYNG9ofjxkzxrjw7T9t2jRDkvHnn39e8hw7duwwJBlxcXF59rVp08aQZMydOzfffW3atLE/3rBhgyHJqFSpkpGenm5vX7JkiSHJmDFjhr2tatWqRnR09BXPec899xh16tS5ZO2G8f9fp4SEhMseZxiGsWzZMkOScfDgQcMwDCM9Pd1wd3c3pk2b5nBcbl9q1aplZGZm2ttnzJhhSDJ++OEHe1unTp2MqlWr5rnW+fPnHZ5rGIZx8uRJIyAgwBgwYIC9LSEhwZBkeHt7G8ePH79iHwzj39evU6dOxvnz543AwEBj3LhxhmEYxv79+w1JxqZNm/J9/7Rv396oV6+ekZGRYW/LyckxmjdvboSHh9vbcp8bGRlp5OTk2NuHDRtmODs7G6mpqfa2OnXqOPzMcmVkZBjZ2dkObQkJCYabm5sxduzYPP3P7/13odyfyeU2T09P+/GnTp0yfH19jUcffdThPMnJyYaPj49D+9mzZ/Nc74MPPjAkGZs3b87zulz4XrtU/wvzGubn1ltvNYKCghyOW7t2rSEpz/vt4vqzsrKMunXrGu3atXNol2S4uroahw4dsrft3bvXkGS89tprl60n9+f08ssvG+fPnzfCw8ONBg0a2PuW+7vnwt810dHReWqVZIwZM+aStRuGYcTHxxuSjPfee8/elvvz37Bhw2XrXLt2reHs7Gw4OzsbERERxtNPP22sWbPGyMrKynPsxb+HGjRoYHTq1Omy569Tp47Rs2dP++NGjRoZPXr0MCQZP/30k2EYhvHxxx8bkoy9e/detp9RUVFG9erVL3s93DiYKoAbgpeX12W/XJT7cdmnn36qnJycq7qGm5ub+vfvX+DjH3roIZUrV87++L777lNQUJA+//zzQl/b19dXv//+u3bs2FHo5+Zn4cKFatKkicLCwiTJ/tFxftMFpH+nT1w4zzR3RPrXX3+94rWcnZ3tz83JydHff/+t8+fPq0mTJvruu+/yHN+9e3dVrFixUP1xdnZWz5499cEHH9j7FxISYq/zQn///be++uor9ezZ0z5qeeLECf3111+KiorSwYMH9ccffzg8Z+DAgQ6jja1atVJ2draOHDlyxdrc3Nzk5PTvr+Ls7Gz99ddf9qkq+fW/oEaPHq1169bl2Tp06OBw3Lp165Samqo+ffrY+3rixAk5OzurWbNmDlM2LvwkICMjQydOnNDtt98uSddUq3R1r+GxY8e0Z88eRUdHy8fHx95+xx13qHbt2nmOv7D+kydPKi0tTa1atcq39sjISPvovCTVr19f3t7eBXpP58oddd27d68++eSTAj8vPxfWfu7cOf31118KCwuTr6/vVb32d9xxh+Lj43X33Xdr7969mjJliqKiolSpUqU8U2Iu5uvrq3379ungwYOXPKZVq1b6+uuvJf07+r93714NHDhQN910k73966+/lq+vr8OI74X9TEtL04kTJ9SmTRv9+uuvSktLK3Q/cf0huOKGcPr0aYeQeLFevXqpRYsWeuSRRxQQEKDevXtryZIlhQqxlSpVKtQXscLDwx0e22w2hYWFXdU6q88884y8vLx02223KTw8XIMGDbJPcyis1NRUff7552rTpo0OHTpk31q0aKGdO3fql19+yfOcKlWqODwuX768pH/DQUG8++67ql+/vn2+XMWKFfXZZ5/l+w9VaGjoVfTq3+kC+/fv1969e7Vo0SL17t0734+2Dx06JMMwNGrUKFWsWNFhGzNmjKS8X165lv7n5ORo2rRpCg8Pl5ubm2666SZVrFhR33///TX9Q12vXj1FRkbm2YKCghyOyw0f7dq1y9PftWvXOvT177//1pNPPqmAgAB5eHioYsWK9p/HtYaKq3kNc0Ptxf8vSdItt9ySp23VqlW6/fbb5e7uLj8/P1WsWFFz5szJt/aL68mtqaDv6Vx9+/ZVWFhYgVcYuJR//vlHo0ePVkhIiMP7JDU19apf+6ZNm+rjjz/WyZMn9e233yomJkanTp3Sfffdd9m542PHjlVqaqpuvvlm1atXTyNHjsyzckurVq107NgxHTp0SFu3bpXNZlNERIRDoP3666/VokUL+x9ukrRlyxZFRkbK09NTvr6+qlixov37AgRXSMxxxQ3g999/V1pamn30MD8eHh7avHmzNmzYoM8++0yrV6/Whx9+qHbt2mnt2rUF+jZ+YeelFsSl5gxmZ2c71FSrVi0dOHBAq1at0urVq/XRRx9p9uzZGj16tF566aVCXXPp0qXKzMzUq6++qldffTXP/oULF+Y556Ven4L8Q/3++++rX79+6tq1q0aOHCl/f385OzsrNjZWhw8fznP81b7OzZo1U40aNTR06FAlJCRccu5v7h8rI0aMuOQXty5+L11L/ydOnKhRo0ZpwIABGjdunPz8/OTk5KShQ4de9eh/YeReY8GCBfnOX75w9Y2ePXtq69atGjlypG699VZ5eXkpJydHd9555zXXei2vYUF8/fXXuvvuu9W6dWvNnj1bQUFBcnFxUVxcXJ4v7hVlPbmjrv369dOnn356VbVL/84BjouL09ChQxURESEfHx/ZbDb17t37ml97V1dXNW3aVE2bNtXNN9+s/v37a+nSpfY/1C7WunVrHT58WJ9++qnWrl2rt956S9OmTdPcuXP1yCOPSJJatmwpSdq8ebN+/fVXNWrUyP7Fy5kzZ+r06dPavXu3JkyYYD/v4cOH1b59e9WsWVNTp05VSEiIXF1d9fnnn2vatGkl8v8DSj+CK657CxYskKRLhpBcTk5Oat++vdq3b6+pU6dq4sSJev7557VhwwZFRkYW+Z22Lv6YzTAMHTp0yP5lBunfEZ7U1NQ8zz1y5IiqV6/u0Obp6alevXqpV69eysrKUrdu3TRhwgTFxMQUaq3ShQsXqm7duvn+o/XGG29o0aJFhQ7D0qVD+LJly1S9enV9/PHHDsdc6h/Na9GnTx+NHz9etWrVyvOlr1y5r6uLi8slb1xwNS7X///85z+aP3++Q3tqaqpuuummIrv+peR+HO7v73/Z/p48eVLr16/XSy+9pNGjR9vbL/dx8YWK4051uWsS51fDxeuifvTRR3J3d9eaNWsclneKi4sr8rou9sADD2j8+PF66aWXdPfdd1/VOZYtW6bo6GiHPyYzMjLy/f1wLXKXFDx27Nhlj/Pz81P//v3Vv39/nT59Wq1bt9aLL75oD65VqlRRlSpV9PXXX+vXX3+1T8tp3bq1hg8frqVLlyo7O9thvdmVK1cqMzNTK1ascBjxvniFEdzYmCqA69pXX32lcePGKTQ0VH379r3kcX///Xeettxgk/stbE9PT0kqsn8o3nvvPYd5t8uWLdOxY8fUsWNHe1uNGjW0bds2ZWVl2dtWrVqVZ9msi5cjcnV1Ve3atWUYhs6dO1fgmpKSkrR582b17NlT9913X56tf//+OnTokH21gMLw9PTM96O+3JGtC0eytm/frvj4+EJf40oeeeQRjRkzJt+R5Fz+/v5q27at3njjjXz/8S7oEk0X8/T0zPe94+zsnGcUb+nSpXnm0RaXqKgoeXt7a+LEifm+V3L7m9/PSVKBbypxqf5fi6CgIN1666169913Hd5b69aty/NRt7Ozs2w2m7Kzs+1tv/322zXPPS2I3FHXPXv2XHH+6OXOcfFr/9prrzn0pzA2bNiQ7+hx7hz7/KZa5Lr4942Xl5fCwsLyrFjRqlUrffXVV/r222/twfXWW29VuXLlNGnSJPuSZLnye4+lpaWVyB8XsA5GXHHd+OKLL/Tzzz/r/PnzSklJ0VdffaV169apatWqWrFixWVHHceOHavNmzerU6dOqlq1qo4fP67Zs2ercuXK9o+8atSoIV9fX82dO1flypWTp6enmjVrdtVzLv38/NSyZUv1799fKSkpmj59usLCwhyW7HrkkUe0bNky3XnnnerZs6cOHz6s999/3+FLI5LUoUMHBQYGqkWLFgoICNBPP/2k119/XZ06dbrs3N6LLVq0SIZhXHJU6K677lKZMmW0cOFCNWvWrFD9bdy4sT788EMNHz5cTZs2lZeXl7p06aLOnTvr448/1r333qtOnTopISFBc+fOVe3atXX69OlCXeNKqlatWqD7v8+aNUstW7ZUvXr19Oijj6p69epKSUlRfHy8fv/996taY7Vx48aaM2eOxo8fr7CwMPn7+6tdu3bq3Lmzxo4dq/79+6t58+b64YcftHDhwjwj6sXF29tbc+bM0YMPPqhGjRqpd+/eqlixohITE/XZZ5+pRYsWev311+Xt7a3WrVtrypQpOnfunCpVqqS1a9cqISGhQNe5VP+vVWxsrDp16qSWLVtqwIAB+vvvv+1rGl/4/unUqZOmTp2qO++8U/fff7+OHz+uWbNmKSwsLM/8zOLQt29fjRs37qpvGd25c2ctWLBAPj4+ql27tuLj4/Xll18WaGmu/AwZMkRnz57Vvffeq5o1ayorK0tbt27Vhx9+qGrVql32i6a1a9dW27Zt1bhxY/n5+Wnnzp1atmyZw7JW0r/BdeHChbLZbPbfo87OzmrevLnWrFmjtm3bOnwvoEOHDnJ1dVWXLl302GOP6fTp05o3b578/f2vOAKMG4gJKxkARSp3OZ3czdXV1QgMDDTuuOMOY8aMGQ5LTuW6eDms9evXG/fcc48RHBxsuLq6GsHBwUafPn2MX375xeF5n376qVG7dm2jTJkyDksTtWnT5pLLUV1qOawPPvjAiImJMfz9/Q0PDw+jU6dOxpEjR/I8/9VXXzUqVapkuLm5GS1atDB27tyZ55xvvPGG0bp1a6NChQqGm5ubUaNGDWPkyJFGWlpantfpcsth1atXz6hSpcol9xuGYbRt29bw9/c3zp07Z+/L0qVLHY7Jb+mm06dPG/fff7/h6+vrsFRRTk6OMXHiRKNq1aqGm5ub0bBhQ2PVqlV5lgi6cJmhgspdDutyLrWc2uHDh42HHnrICAwMNFxcXIxKlSoZnTt3NpYtW3bF5+a3JFFycrLRqVMno1y5coYk+88vIyPDeOqpp4ygoCDDw8PDaNGihREfH5/nZ1zY5bAu/pnkio6OdlgO68LnRUVFGT4+Poa7u7tRo0YNo1+/fsbOnTvtx/z+++/Gvffea/j6+ho+Pj5Gjx49jKNHj+ZZuim/99ql+l+Y1/BSPvroI6NWrVqGm5ubUbt2bePjjz/Od4mp+fPnG+Hh4Yabm5tRs2ZNIy4uLs/vAsP4dymqQYMG5bnOpZanu9Dl3qcX/q4q7HJYJ0+eNPr372/cdNNNhpeXlxEVFWX8/PPPeWoq6Ov2xRdfGAMGDDBq1qxpeHl5Ga6urkZYWJgxZMgQIyUl5bL9Hj9+vHHbbbcZvr6+hoeHh1GzZk1jwoQJeZbS2rdvn325vAuNHz/ekGSMGjUqT10rVqww6tevb7i7uxvVqlUzJk+ebLz99tsFXsoP1z+bYRTRzHcAAACgGDHHFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlXPc3IMjJydHRo0dVrly5YrnlIAAAAK6NYRg6deqUgoOD5eR06XHV6z64Hj16VCEhIWaXAQAAgCtISkpS5cqVL7n/ug+uube7TEpKkre3t8nVAAAA4GLp6ekKCQm54m3Kr/vgmjs9wNvbm+AKAABQil1pWqepX87Kzs7WqFGjFBoaKg8PD9WoUUPjxo3ThXehNQxDo0ePVlBQkDw8PBQZGamDBw+aWDUAAADMYGpwnTx5subMmaPXX39dP/30kyZPnqwpU6botddesx8zZcoUzZw5U3PnztX27dvl6empqKgoZWRkmFg5AAAASprNuHB4s4R17txZAQEBmj9/vr2te/fu8vDw0Pvvvy/DMBQcHKynnnpKI0aMkCSlpaUpICBA77zzjnr37n3Fa6Snp8vHx0dpaWlMFQAAACiFCprXTJ3j2rx5c7355pv65ZdfdPPNN2vv3r365ptvNHXqVElSQkKCkpOTFRkZaX+Oj4+PmjVrpvj4+HyDa2ZmpjIzM+2P09PTr1iHYRg6f/68srOzi6BXsCpnZ2eVKVOGZdMAACilTA2uzz77rNLT01WzZk05OzsrOztbEyZMUN++fSVJycnJkqSAgACH5wUEBNj3XSw2NlYvvfRSgWvIysrSsWPHdPbs2avsBa4nZcuWVVBQkFxdXc0uBQAAXMTU4LpkyRItXLhQixYtUp06dbRnzx4NHTpUwcHBio6OvqpzxsTEaPjw4fbHucsr5CcnJ0cJCQlydnZWcHCwXF1dGW27QRmGoaysLP35559KSEhQeHj4ZRdABgAAJc/U4Dpy5Eg9++yz9o/869WrpyNHjig2NlbR0dEKDAyUJKWkpCgoKMj+vJSUFN166635ntPNzU1ubm4Fun5WVpZycnIUEhKismXLXltnYHkeHh5ycXHRkSNHlJWVJXd3d7NLAgAAFzB1SOns2bN5RrWcnZ2Vk5MjSQoNDVVgYKDWr19v35+enq7t27crIiKiyOpgZA25eC8AAFB6mTri2qVLF02YMEFVqlRRnTp1tHv3bk2dOlUDBgyQ9O8itEOHDtX48eMVHh6u0NBQjRo1SsHBweratauZpQMAAKCEmRpcX3vtNY0aNUqPP/64jh8/ruDgYD322GMaPXq0/Zinn35aZ86c0cCBA5WamqqWLVtq9erVfIwLAABwgzF1HdeScLl1wTIyMpSQkKDQ0FCCMCTxngAAwAyWWMe1tKr27Gcler3fJnUq0esBAABYEd9EwXUpKyvL7BIAAEARI7haVE5OjmJjYxUaGioPDw81aNBAy5Ytk2EYioyMVFRUlHJngfz999+qXLmyfe5wdna2Hn74Yftzb7nlFs2YMcPh/P369VPXrl01ceJEBQQEyNfXV2PHjtX58+c1cuRI+fn5qXLlyoqLiytQvVlZWRo8eLCCgoLk7u6uqlWrKjY21r4/NTVVjz32mAICAuTu7q66detq1apV9v0fffSR6tSpIzc3N1WrVk2vvvqqw/mrVaumcePG6aGHHpK3t7cGDhwoSfrmm2/UqlUreXh4KCQkRE888YTOnDlT+BccAACYjqkCFhUbG6v3339fc+fOVXh4uDZv3qwHHnhAFStW1Lvvvqt69epp5syZevLJJ/Xf//5XlSpVsgfXnJwcVa5cWUuXLlWFChW0detWDRw4UEFBQerZs6f9Gl999ZUqV66szZs3a8uWLXr44Ye1detWtW7dWtu3b9eHH36oxx57THfccYcqV6582XpnzpypFStWaMmSJapSpYqSkpKUlJRkr6djx446deqU3n//fdWoUUP79++Xs7OzJGnXrl3q2bOnXnzxRfXq1Utbt27V448/rgoVKqhfv372a7zyyisaPXq0xowZI0k6fPiw7rzzTo0fP15vv/22/vzzTw0ePFiDBw8ucOAGAAClB1/OyueLOKV9jmtmZqb8/Pz05ZdfOqxn+8gjj+js2bNatGiRli5dqoceekhDhw7Va6+9pt27dys8PPyS5xw8eLCSk5O1bNkySf+OuG7cuFG//vqrfW3TmjVryt/fX5s3b5b078itj4+P3nrrLftNJC7liSee0L59+/Tll1/muTvZ2rVr1bFjR/3000+6+eab8zy3b9+++vPPP7V27Vp729NPP63PPvtM+/btk/TviGvDhg21fPlyh9fD2dlZb7zxhr3tm2++UZs2bXTmzJl8v3zFl7MAAFZW0hnmQtfynR2+nHUdO3TokM6ePas77rjDoT0rK0sNGzaUJPXo0UPLly/XpEmTNGfOnDyhddasWXr77beVmJiof/75R1lZWXnuRlanTh2HBfkDAgJUt25d+2NnZ2dVqFBBx48fv2LN/fr10x133KFbbrlFd955pzp37qwOHTpIkvbs2aPKlSvnG1ol6aefftI999zj0NaiRQtNnz5d2dnZ9pHZJk2aOByzd+9eff/991q4cKG9zTAM+61+a9WqdcW6AQBA6UFwtaDTp09Lkj777DNVqlTJYV/u7W7Pnj2rXbt2ydnZWQcPHnQ4ZvHixRoxYoReffVVRUREqFy5cnr55Ze1fft2h+NcXFwcHttstnzbcu90djmNGjVSQkKCvvjiC3355Zfq2bOnIiMjtWzZMnl4eBSs41fg6enp8Pj06dN67LHH9MQTT+Q5tkqVKkVyTQAAUHIIrhZUu3Ztubm5KTExUW3atMn3mKeeekpOTk764osvdNddd6lTp05q166dJGnLli1q3ry5Hn/8cfvxhw8fLva6vb291atXL/Xq1Uv33Xef7rzzTv3999+qX7++fv/9d/3yyy/5jrrWqlVLW7ZscWjbsmWLbr75Zvtoa34aNWqk/fv3KywsrMj7AgAASh7B1YLKlSunESNGaNiwYcrJyVHLli2VlpamLVu2yNvbWzfddJPefvttxcfHq1GjRho5cqSio6P1/fffq3z58goPD9d7772nNWvWKDQ0VAsWLNCOHTsUGhpabDVPnTpVQUFBatiwoZycnLR06VIFBgbK19dXbdq0UevWrdW9e3dNnTpVYWFh+vnnn2Wz2XTnnXfqqaeeUtOmTTVu3Dj16tVL8fHxev311zV79uzLXvOZZ57R7bffrsGDB+uRRx6Rp6en9u/fr3Xr1un1118vtr4CAIDiQXDNhxVuCDBu3DhVrFhRsbGx+vXXX+Xr66tGjRopJiZGvXr10osvvqhGjRpJkl566SWtXbtW//3vf+0rAezevVu9evWSzWZTnz599Pjjj+uLL74otnrLlSunKVOm6ODBg3J2dlbTpk31+eef2+fQfvTRRxoxYoT69OmjM2fOKCwsTJMmTZL078jpkiVLNHr0aI0bN05BQUEaO3asw4oC+alfv742bdqk559/Xq1atZJhGKpRo4Z69epVbP0EAJQOVv2SEi6PVQX4BjkuwHsCAK4PN2pwtWq/C7qqADcgAAAAgCUQXFEkJk6cKC8vr3y3jh07ml0eAAC4DjDHFUXiv//9r8Ndty5UVMtdAQCAGxvBFUXCz89Pfn5+ZpcBAACuY0wV0L93UwIk3gsAAJRmN3Rwzb0L1NmzZ02uBKVF7nvh4juEAQAA893QUwWcnZ3l6+ur48ePS5LKli0rm81mclUwg2EYOnv2rI4fPy5fX9/L3pELAACY44YOrpIUGBgoSfbwihubr6+v/T0BAABKlxs+uNpsNgUFBcnf31/nzp0zuxyYyMXFhZFWAABKsRs+uOZydnYmtAAArjtWvZMSkJ8b+stZAAAAsA5GXAEANwRGHgHrY8QVAAAAlkBwBQAAgCUQXAEAAGAJBFcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJBFcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJBFcAAABYAsEVAAAAllDG7AIAACWr2rOfmXbt3yZ1Mu3aAKzP1BHXatWqyWaz5dkGDRokScrIyNCgQYNUoUIFeXl5qXv37kpJSTGzZAAAAJjE1OC6Y8cOHTt2zL6tW7dOktSjRw9J0rBhw7Ry5UotXbpUmzZt0tGjR9WtWzczSwYAAIBJTJ0qULFiRYfHkyZNUo0aNdSmTRulpaVp/vz5WrRokdq1aydJiouLU61atbRt2zbdfvvtZpQMAAAAk5SaL2dlZWXp/fff14ABA2Sz2bRr1y6dO3dOkZGR9mNq1qypKlWqKD4+/pLnyczMVHp6usMGAAAA6ys1wfWTTz5Ramqq+vXrJ0lKTk6Wq6urfH19HY4LCAhQcnLyJc8TGxsrHx8f+xYSElKMVQMAAKCklJrgOn/+fHXs2FHBwcHXdJ6YmBilpaXZt6SkpCKqEAAAAGYqFcthHTlyRF9++aU+/vhje1tgYKCysrKUmprqMOqakpKiwMDAS57Lzc1Nbm5uxVkuAAAATFAqRlzj4uLk7++vTp3+//p+jRs3louLi9avX29vO3DggBITExUREWFGmQAAADCR6SOuOTk5iouLU3R0tMqU+f/l+Pj46OGHH9bw4cPl5+cnb29vDRkyRBEREawoAAAAcAMyPbh++eWXSkxM1IABA/LsmzZtmpycnNS9e3dlZmYqKipKs2fPNqFKAAAAmM304NqhQwcZhpHvPnd3d82aNUuzZs0q4aoAAABQ2pSKOa4AAADAlRBcAQAAYAkEVwAAAFgCwRUAAACWQHAFAACAJRBcAQAAYAkEVwAAAFgCwRUAAACWQHAFAACAJZh+5ywAMEu1Zz8z7dq/Tepk2rUBwKoYcQUAAIAlEFwBAABgCUwVAMBH5gAAS2DEFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJbAnbOAC3AHKQAASi9GXAEAAGAJBFcAAABYAlMFkC8+MgcAAKUNI64AAACwBEZcr4CRRwAAgNKBEVcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJBFcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJpgfXP/74Qw888IAqVKggDw8P1atXTzt37rTvNwxDo0ePVlBQkDw8PBQZGamDBw+aWDEAAADMYGpwPXnypFq0aCEXFxd98cUX2r9/v1599VWVL1/efsyUKVM0c+ZMzZ07V9u3b5enp6eioqKUkZFhYuUAAAAoaabe8nXy5MkKCQlRXFycvS00NNT+34ZhaPr06XrhhRd0zz33SJLee+89BQQE6JNPPlHv3r1LvGYAAACYw9QR1xUrVqhJkybq0aOH/P391bBhQ82bN8++PyEhQcnJyYqMjLS3+fj4qFmzZoqPj8/3nJmZmUpPT3fYAAAAYH2mBtdff/1Vc+bMUXh4uNasWaP//e9/euKJJ/Tuu+9KkpKTkyVJAQEBDs8LCAiw77tYbGysfHx87FtISEjxdgIAAAAlwtTgmpOTo0aNGmnixIlq2LChBg4cqEcffVRz58696nPGxMQoLS3NviUlJRVhxQAAADCLqcE1KChItWvXdmirVauWEhMTJUmBgYGSpJSUFIdjUlJS7Psu5ubmJm9vb4cNAAAA1mdqcG3RooUOHDjg0PbLL7+oatWqkv79olZgYKDWr19v35+enq7t27crIiKiRGsFAACAuUxdVWDYsGFq3ry5Jk6cqJ49e+rbb7/Vm2++qTfffFOSZLPZNHToUI0fP17h4eEKDQ3VqFGjFBwcrK5du5pZOgAAAEqYqcG1adOmWr58uWJiYjR27FiFhoZq+vTp6tu3r/2Yp59+WmfOnNHAgQOVmpqqli1bavXq1XJ3dzexcgAAAJQ0U4OrJHXu3FmdO3e+5H6bzaaxY8dq7NixJVgVAAAAShvTb/kKAAAAFATBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCQRXAAAAWALBFQAAAJZAcAUAAIAlEFwBAABgCaYG1xdffFE2m81hq1mzpn1/RkaGBg0apAoVKsjLy0vdu3dXSkqKiRUDAADALKaPuNapU0fHjh2zb998841937Bhw7Ry5UotXbpUmzZt0tGjR9WtWzcTqwUAAIBZypheQJkyCgwMzNOelpam+fPna9GiRWrXrp0kKS4uTrVq1dK2bdt0++23l3SpAAAAMJHpI64HDx5UcHCwqlevrr59+yoxMVGStGvXLp07d06RkZH2Y2vWrKkqVaooPj7+kufLzMxUenq6wwYAAADrMzW4NmvWTO+8845Wr16tOXPmKCEhQa1atdKpU6eUnJwsV1dX+fr6OjwnICBAycnJlzxnbGysfHx87FtISEgx9wIAAAAlwdSpAh07drT/d/369dWsWTNVrVpVS5YskYeHx1WdMyYmRsOHD7c/Tk9PJ7wCAABcB0yfKnAhX19f3XzzzTp06JACAwOVlZWl1NRUh2NSUlLynROby83NTd7e3g4bAAAArK9UBdfTp0/r8OHDCgoKUuPGjeXi4qL169fb9x84cECJiYmKiIgwsUoAAACYwdSpAiNGjFCXLl1UtWpVHT16VGPGjJGzs7P69OkjHx8fPfzwwxo+fLj8/Pzk7e2tIUOGKCIighUFAAAAbkCmBtfff/9dffr00V9//aWKFSuqZcuW2rZtmypWrChJmjZtmpycnNS9e3dlZmYqKipKs2fPNrNkAAAAmMTU4Lp48eLL7nd3d9esWbM0a9asEqoIAAAApVWpmuMKAAAAXMpVjbieOXNGmzZtUmJiorKyshz2PfHEE0VSGAAAAHChQgfX3bt366677tLZs2d15swZ+fn56cSJEypbtqz8/f0JrgAAACgWhZ4qMGzYMHXp0kUnT56Uh4eHtm3bpiNHjqhx48Z65ZVXiqNGAAAAoPDBdc+ePXrqqafk5OQkZ2dnZWZmKiQkRFOmTNFzzz1XHDUCAAAAhQ+uLi4ucnL692n+/v5KTEyUJPn4+CgpKaloqwMAAAD+T6HnuDZs2FA7duxQeHi42rRpo9GjR+vEiRNasGCB6tatWxw1AgAAAIUfcZ04caKCgoIkSRMmTFD58uX1v//9T3/++afefPPNIi8QAAAAkK5ixLVJkyb2//b399fq1auLtCAAAAAgP4UecX377beVkJBQHLUAAAAAl1To4BobG6uwsDBVqVJFDz74oN566y0dOnSoOGoDAAAA7AodXA8ePKjExETFxsaqbNmyeuWVV3TLLbeocuXKeuCBB4qjRgAAAKDwwVWSKlWqpL59+2ratGmaMWOGHnzwQaWkpGjx4sVFXR8AAAAg6Sq+nLV27Vpt3LhRGzdu1O7du1WrVi21adNGy5YtU+vWrYujRgAAAKDwwfXOO+9UxYoV9dRTT+nzzz+Xr69vMZQFAAAAOCr0VIGpU6eqRYsWmjJliurUqaP7779fb775pn755ZfiqA8AAACQdBXBdejQofr444914sQJrV69Ws2bN9fq1atVt25dVa5cuThqBAAAAAo/VUCSDMPQ7t27tXHjRm3YsEHffPONcnJyVLFixaKuDwAAAJB0FcG1S5cu2rJli9LT09WgQQO1bdtWjz76qFq3bs18VwAAABSbQgfXmjVr6rHHHlOrVq3k4+NTHDUBAAAAeRQ6uL788st52lJTUxltBQAAQLEq9JezJk+erA8//ND+uGfPnvLz81OlSpW0d+/eIi0OAAAAyFXo4Dp37lyFhIRIktatW6d169Zp9erV6tixo0aOHFnkBQIAAADSVUwVSE5OtgfXVatWqWfPnurQoYOqVaumZs2aFXmBAAAAgHQVI67ly5dXUlKSJGn16tWKjIyU9O8SWdnZ2UVbHQAAAPB/Cj3i2q1bN91///0KDw/XX3/9pY4dO0qSdu/erbCwsCIvEAAAAJCuIrhOmzZN1apVU1JSkqZMmSIvLy9J0rFjx/T4448XeYEAAACAdBXB1cXFRSNGjMjTPmzYsCIpCAAAAMhPgYLrihUr1LFjR7m4uGjFihWXPfbuu+8uksIAAACACxUouHbt2lXJycny9/dX165dL3mczWbjC1oAAAAoFgUKrjk5Ofn+NwAAAFBSCr0cFgAAAGCGAgfX+Ph4rVq1yqHtvffeU2hoqPz9/TVw4EBlZmYWeYEAAACAVIjgOnbsWO3bt8/++IcfftDDDz+syMhIPfvss1q5cqViY2OLpUgAAACgwMF1z549at++vf3x4sWL1axZM82bN0/Dhw/XzJkztWTJkmIpEgAAAChwcD158qQCAgLsjzdt2mS/a5YkNW3a1H4rWAAAAKCoFTi4BgQEKCEhQZKUlZWl7777Trfffrt9/6lTp+Ti4nLVhUyaNEk2m01Dhw61t2VkZGjQoEGqUKGCvLy81L17d6WkpFz1NQAAAGBdBQ6ud911l5599ll9/fXXiomJUdmyZdWqVSv7/u+//141atS4qiJ27NihN954Q/Xr13doHzZsmFauXKmlS5dq06ZNOnr0qLp163ZV1wAAAIC1FTi4jhs3TmXKlFGbNm00b948zZs3T66urvb9b7/9tjp06FDoAk6fPq2+fftq3rx5Kl++vL09LS1N8+fP19SpU9WuXTs1btxYcXFx2rp1q7Zt21bo6wAAAMDaCnQDAkm66aabtHnzZqWlpcnLy0vOzs4O+5cuXSovL69CFzBo0CB16tRJkZGRGj9+vL19165dOnfunCIjI+1tNWvWVJUqVRQfH+8wTeFCmZmZDstypaenF7omAAAAlD4FDq65fHx88m338/Mr9MUXL16s7777Tjt27MizLzk5Wa6urvL19XVoDwgIUHJy8iXPGRsbq5deeqnQtQAAAKB0M+3OWUlJSXryySe1cOFCubu7F9l5Y2JilJaWZt9Y6QAAAOD6YFpw3bVrl44fP65GjRqpTJkyKlOmjDZt2qSZM2eqTJkyCggIUFZWllJTUx2el5KSosDAwEue183NTd7e3g4bAAAArK/QUwWKSvv27fXDDz84tPXv3181a9bUM888o5CQELm4uGj9+vXq3r27JOnAgQNKTExURESEGSUDAADARKYF13Llyqlu3boObZ6enqpQoYK9/eGHH9bw4cPl5+cnb29vDRkyRBEREZf8YhYAAACuXwUKritWrFDHjh3l4uKiFStWXPbYu+++u0gKk6Rp06bJyclJ3bt3V2ZmpqKiojR79uwiOz8AAACso0DBtWvXrkpOTpa/v7+6du16yeNsNpuys7OvupiNGzc6PHZ3d9esWbM0a9asqz4nAAAArg8FCq45OTn5/jcAAABQUkxbVQAAAAAojKv6ctaZM2e0adMmJSYmKisry2HfE088USSFAQAAABcqdHDdvXu37rrrLp09e1ZnzpyRn5+fTpw4obJly8rf35/gCgAAgGJR6KkCw4YNU5cuXXTy5El5eHho27ZtOnLkiBo3bqxXXnmlOGoEAAAACh9c9+zZo6eeekpOTk5ydnZWZmamQkJCNGXKFD333HPFUSMAAABQ+ODq4uIiJ6d/n+bv76/ExERJko+Pj5KSkoq2OgAAAOD/FHqOa8OGDbVjxw6Fh4erTZs2Gj16tE6cOKEFCxbkuRMWAAAAUFQKPeI6ceJEBQUFSZImTJig8uXL63//+5/+/PNPvfnmm0VeIAAAACBdxYhrkyZN7P/t7++v1atXF2lBAAAAQH64AQEAAAAsoUAjrg0bNpTNZivQCb/77rtrKggAAADIT4GCa9euXYu5DAAAAODyChRcx4wZU9x1AAAAAJfFHFcAAABYQoFGXP38/PTLL7/opptuUvny5S873/Xvv/8usuIAAACAXAUKrtOmTVO5cuUkSdOnTy/OegAAAIB8FSi4RkdH5/vfAAAAQElhjisAAAAsocB3znJycrriWq42m03nz5+/5qIAAACAixU4uC5fvvyS++Lj4zVz5kzl5OQUSVEAAADAxQocXO+55548bQcOHNCzzz6rlStXqm/fvho7dmyRFgcAAADkuqo5rkePHtWjjz6qevXq6fz589qzZ4/effddVa1atajrAwAAACQVMrimpaXpmWeeUVhYmPbt26f169dr5cqVqlu3bnHVBwAAAEgqxFSBKVOmaPLkyQoMDNQHH3yQ79QBAAAAoLgUOLg+++yz8vDwUFhYmN599129++67+R738ccfF1lxAAAAQK4CB9eHHnroisthAQAAAMWlwMH1nXfeKcYyAAAAgMvjzlkAAACwBIIrAAAALIHgCgAAAEsg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yLO03S+//KKqVauaVFHJi4uLk7+/vzp16mR2KcXu7NmzcnJyjAfOzs7KyckxqaKS5+npqaCgIJ08eVJr1qzRPffcY3ZJJSY0NFSBgYFav369vS09PV3bt28vFd/dYKpAMTh9+rTD6EtCQoL27NkjPz8/ValSxcTKitegQYO0aNEiffrppypXrpx9LoyPj488PDxMrq74xMTEqGPHjqpSpYpOnTqlRYsWaePGjVqzZo3ZpRWbcuXK5Zm77OnpqQoVKlz3c5pHjBihLl26qGrVqjp69KjGjBkjZ2dn9enTx+zSis2wYcPUvHlzTZw4UT179tS3336rN998U2+++abZpZWInJwcxcXFKTo6WmXKXP//bHbp0kUTJkxQlSpVVKdOHe3evVtTp07VgAEDzC6t2K1Zs0aGYeiWW27RoUOHNHLkSNWsWVP9+/c3u7QidaWcMnToUI0fP17h4eEKDQ3VqFGjFBwcrK5du5pXdC6zlzW4Hm3YsMGQlGeLjo42u7RilV+fJRlxcXFml1asBgwYYFStWtVwdXU1KlasaLRv395Yu3at2WWVuBtlOaxevXoZQUFBhqurq1GpUiWjV69exqFDh8wuq9itXLnSqFu3ruHm5mbUrFnTePPNN80uqcSsWbPGkGQcOHDA7FJKRHp6uvHkk08aVapUMdzd3Y3q1asbzz//vJGZmWl2acXuww8/NKpXr264uroagYGBxqBBg4zU1FSzyypyV8opOTk5xqhRo4yAgADDzc3NaN++fal5/9sM4wa4FQYAAAAsjzmuAAAAsASCKwAAACyB4AoAAABLILgCAADAEgiuAAAAsASCKwAAACyB4AoAAABLILgCAADAEgiuAGBRGzdulM1mU2pqqtmlAECJILgCQDHr16+fbDabbDabXFxcFBoaqqeffloZGRkFPkfbtm01dOhQh7bmzZvr2LFj8vHxKeKKAaB0KmN2AQBwI7jzzjsVFxenc+fOadeuXYqOjpbNZtPkyZOv+pyurq4KDAwswioBoHRjxBUASoCbm5sCAwMVEhKirl27KjIyUuvWrZMk/fXXX+rTp48qVaqksmXLql69evrggw/sz+3Xr582bdqkGTNm2Eduf/vttzxTBd555x35+vpqzZo1qlWrlry8vHTnnXfq2LFj9nOdP39eTzzxhHx9fVWhQgU988wzio6OVteuXUvy5QCAq0JwBYAS9uOPP2rr1q1ydXWVJGVkZKhx48b67LPP9OOPP2rgwIF68MEH9e2330qSZsyYoYiICD366KM6duyYjh07ppCQkHzPffbsWb3yyitasGCBNm/erMTERI0YMcK+f/LkyVq4cKHi4uK0ZcsWpaen65NPPin2PgNAUWCqAACUgFWrVsnLy0vnz59XZmamnJyc9Prrr0uSKlWq5BAuhwwZojVr1mjJkiW67bbb5OPjI1dXV5UtW/aKUwPOnTunuXPnqkaNGpKkwYMHa+zYsfb9r732mmJiYnTvvfdKkl5//XV9/vnnRd1dACgWBFcAKAH/+c9/NGfOHJ05c0bTpk1TmTJl1L17d0lSdna2Jk6cqCVLluiPP/5QVlaWMjMzVbZs2UJfp2zZsvbQKklBQUE6fvy4JCktLU0pKSm67bbb7PudnZ3VuHFj5eTkXGMPAaD4MVUAAEqAp6enwsLC1KBBA7399tvavn275s+fL0l6+eWXNWPGDD3zzDPasGGD9uzZo6ioKGVlZRX6Oi4uLg6PbTabDMMokj4AgNkIrgBQwpycnPTcc8/phRde0D///KMtW7bonnvu0QMPPKAGDRqoevXq+uWXXxye4+rqquzs7Gu6ro+PjwICArRjxw57W3Z2tr777rtrOi8AlBSCKwCYoEePHnJ2dtasWbMUHh6udevWaevWrfrpp5/02GOPKSUlxeH4atWqafv27frtt9904sSJq/5of8iQIYqNjdWnn36qAwcO6Mknn9TJkydls9mKolsAUKwIrgBggjJlymjw4MGaMmWKnnrqKTVq1EhRUVFq27atAgMD8yxPNWLECDk7O6t27dqqWLGiEhMTr+q6zzzzjPr06aOHHnpIERER8vLyUlRUlNzd3YugVwBQvGwGk58A4IaVk5OjWrVqqWfPnho3bpzZ5QDAZbGqAADcQI4cOaK1a9eqTZs2yszM1Ouvv66EhATdf//9ZpcGAFfEVAEAuIE4OTnpnXfeUdOmTdWiRQv98MMP+vLLL1WrVi2zSwOAK2KqAAAAACyBEVcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJBFcAAABYAsEVAAAAlkBwBQAAgCUQXAEAAGAJ/w9KoBMY+81YPgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Insight:**\n"
      ],
      "metadata": {
        "id": "0sy8wipQ7YM-"
      }
    },
    {
      "source": [
        "sleep = df.groupby('sleep_hours')[['exam_score']].mean()\n",
        "\n",
        "sleep = sleep.reset_index()\n",
        "\n",
        "sleep['sleep_bin'] = pd.cut(sleep['sleep_hours'],\n",
        "                            bins=[0, 4.99, 6.99, 8.99, 24],\n",
        "                            labels=['<5', '5–6', '7–8', '9+'])\n",
        "\n",
        "grouped = sleep.groupby('sleep_bin')['exam_score'].mean().reset_index()\n",
        "\n",
        "print(grouped)"
      ],
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yZk-AAW08lJT",
        "outputId": "5988764a-a311-4568-b768-22f0ddd4e419"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  sleep_bin  exam_score\n",
            "0        <5   63.447909\n",
            "1       5–6   68.758102\n",
            "2       7–8   71.412745\n",
            "3        9+   69.835185\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-18-428d4752ea41>:9: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.\n",
            "  grouped = sleep.groupby('sleep_bin')['exam_score'].mean().reset_index()\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(8, 5))\n",
        "plt.bar(grouped['sleep_bin'], grouped['exam_score'], color='mediumseagreen')\n",
        "plt.title('Rata-rata Nilai berdasarkan Kelompok Lama Tidur')\n",
        "plt.xlabel('Kelompok Lama Tidur (jam)')\n",
        "plt.ylabel('Rata-rata Nilai')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "fOMSmdLT70uf",
        "outputId": "c40543ae-a5cf-429f-91b9-d163ef922be9"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ],
            "image/png": "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/vmnce+99xqSDEdHR6NRo0ZGr169jBo1ahiGkfX8slgshouLS5ZzDYWHn3xFrk6fPq0PPvhACxcuVFxcnCIiItSnTx+1atVKDg65jzTZv3+/6tSpowMHDqh69ep3qGKYRaNGjRQWFqaoqCglJiZmWV6lShVt2LBBrVu31g8//KDmzZtblzVu3FhhYWH63//+dydLRhFz/PhxVapUSUuXLlXnzp0lSWfPns3xfFq3bp3atm2rixcvytPT07qsatWqGjhwoF588cU7VjuKtitXrigxMVFly5bVI488oqSkJK1cuTLL+dWqVStNnjxZjRs3trZxdbZwMVQAuSpXrpzGjBmjMWPGaMuWLVqwYIG6desmDw8P9erVS3369FHt2rWz3Xb37t3Wj+KA6yUlJenw4cPq06ePypQpk+M5EhISImdnZx08eNAaXK9du6Zjx44pMDDwTpaMIigmJkZlypRRhw4drG25nU9Xr16VpCz/6XZwcFBGRkbhFQrTcXNzk5ubmy5evKjVq1drypQpkrKeX8WKFdNdd91FWL2DCK7Is6ZNm6pp06aaPn26li1bpvnz5+vNN9/Url27lJSUpO3bt+u+++6Th4eHtm7dqqioKPXu3ZuxY9DIkSPVqVMnBQYG6vTp0xo7dqwcHR3Vs2fPXLfz9PTUk08+qbFjxyogIECBgYF64403JOXt1kf498rIyFBMTIz69eunYsXy9k9ZaGiovL291a9fP73yyitydXXVe++9p6NHj9qEX/x3rV69WoZhqHr16vrzzz/13HPPqUaNGurfv7+9S8P/R3BFvrm4uKhHjx7q0aOHTp8+LXd3d/35559avHixXn31VaWkpCgoKEhRUVEaMWKEvctFEXDy5En17NlTf/31l3x9fdW8eXNt27Yty22NsvPGG2+oWLFi6tOnj/7++281btxY69ev5z9E/3Fr165VbGysBgwYkOdtSpcurVWrVmn06NFq3bq1rl27ptq1a2v58uU2d0fBf9elS5c0atQonTx5Uj4+PoqIiND//vc/FS9e3N6l4f9jjCsAAABMgfu4AgAAwBQIrgAAADAFgisAAABMgeAKAAAAUyC4AgAAwBQIrgAAADAFgisAAABMgeAKAAAAUyC4ArAri8WiZcuW2buMPGnVqpWGDx9u7zLsYuPGjbJYLEpISMhxnfnz56tkyZKFVkNqaqqqVKmiLVu2WNvuxPmTmpqqihUr6ueffy7U/QC4OYIrgFsWGRmpLl262LR9/vnncnFx0VtvvWWfooqQY8eOyWKxaPfu3fYuJVeRkZGyWCw5ThUrVlTTpk115swZeXl52a3O2bNnKygoSE2bNrW2nTlzRu3atSvU/To5OWnkyJF64YUXCnU/AG6O4AqgwLz//vvq1auXZs2apWeffdbe5SCPpk+frjNnzlgnSYqJibHO79ixQ05OTvL395fFYinUWlJTU7NtNwxDM2bM0MCBA23a/f395ezsXKg1SVKvXr30448/av/+/YW+LwA5I7gCKBBTpkzR0KFDtXjxYvXv39/avnz5cjVs2FAuLi6qVKmSxo0bp7S0tBz72bt3r1q3bi1XV1eVKlVKTzzxhJKSkqzLM6/yTpgwQX5+fipZsqTGjx+vtLQ0Pffcc/Lx8VH58uUVExNj3SbzyufixYvVtGlTubi4qE6dOtq0aZPNvjdt2qR77rlHzs7OKlu2rF588cVca125cqW8vLy0aNGiWzlkOnz4sDp37iw/Pz+5u7urUaNGWrt2rc06FStW1Ouvv66+ffvK3d1dgYGB+uqrr3Tu3Dl17txZ7u7uCg4OtvkY+6+//lLPnj111113qUSJEqpbt64++eSTHOvw8vKSv7+/dZKkkiVLWud9fX2zHSowf/58VahQQSVKlFDXrl31119/2fSb3RX54cOHq1WrVtb5Vq1aaciQIRo+fLhKly6t8PDwbGvcuXOnDh8+rA4dOti03zhU4IUXXlC1atVUokQJVapUSS+//LKuXbtmXf7qq6+qfv36+uCDD1ShQgW5u7vr6aefVnp6uqZMmSJ/f3+VKVNG//vf/2z24+3trWbNmmnx4sU5HkcAhY/gCuC2vfDCC3rttdf09ddfq2vXrtb2H374QX379tWwYcP022+/ac6cOZo/f36WUJDpypUrCg8Pl7e3t3bs2KElS5Zo7dq1GjJkiM1669ev1+nTp/X9999r6tSpGjt2rDp27Chvb29t375dTz75pAYNGqSTJ0/abPfcc8/p2Wef1a5duxQaGqpOnTpZw9apU6fUvn17NWrUSHv27NGsWbM0b948vf7669nW+vHHH6tnz55atGiRevXqdUvHLSkpSe3bt9e6deu0a9cuPfDAA+rUqZNiY2Nt1ps2bZqaNWumXbt2qUOHDurTp4/69u2r3r1765dfflHlypXVt29fGYYhSUpOTlZISIhWrlypffv26YknnlCfPn30008/3VKd2dm+fbsGDhyoIUOGaPfu3brvvvtyPFY3s2DBAjk5OWnz5s2aPXt2tuv88MMPqlatmjw8PHLty8PDQ/Pnz9dvv/2m6dOn67333tO0adNs1jl8+LC+/fZbrVq1Sp988onmzZunDh066OTJk9q0aZMmT56sMWPGaPv27Tbb3XPPPfrhhx9u6TECKCAGANyifv36GU5OToYkY926dVmWt2nTxpgwYYJN24cffmiULVvWOi/J+PLLLw3DMIy5c+ca3t7eRlJSknX5ypUrDQcHByMuLs66z8DAQCM9Pd26TvXq1Y0WLVpY59PS0gw3Nzfjk08+MQzDMI4ePWpIMiZNmmRd59q1a0b58uWNyZMnG4ZhGC+99JJRvXp1IyMjw7pOdHS04e7ubt1Xy5YtjWHDhhkzZswwvLy8jI0bN+Z6fDL3u2vXrlzXu17t2rWNd9991zofGBho9O7d2zp/5swZQ5Lx8ssvW9u2bt1qSDLOnDmTY78dOnQwnn322TzVcP1zkmnDhg2GJOPixYuGYRhGz549jfbt29us88gjjxheXl7W+X79+hmdO3e2WWfYsGFGy5YtrfMtW7Y0GjRocNOahg0bZrRu3TpPtV7vjTfeMEJCQqzzY8eONUqUKGEkJiZa28LDw42KFStmOacmTpxo09f06dONihUr3rRWAIWnmN0SM4B/heDgYJ0/f15jx47VPffcI3d3d+uyPXv2aPPmzTZXWNPT05WcnKyrV6+qRIkSNn39/vvvqlevntzc3KxtzZo1U0ZGhg4ePCg/Pz9JUu3ateXg8H8fGPn5+alOnTrWeUdHR5UqVUpnz5616T80NNT6d7FixXT33Xfr999/t+47NDTUZgxns2bNlJSUpJMnT6pChQqS/vny2dmzZ7V582Y1atQo/wfsOklJSXr11Ve1cuVKnTlzRmlpafr777+zXHENDg62eaySVLdu3SxtZ8+elb+/v9LT0zVhwgR99tlnOnXqlFJTU5WSkpLleN+O33//3ebquvTP8V21alW++woJCbnpOn///bdcXFxuut6nn36qd955R4cPH1ZSUpLS0tLk6elps07FihVtrtz6+fnJ0dExyzl14/nj6uqqq1ev3rQGAIWHoQIAbstdd92ljRs36tSpU3rggQd0+fJl67KkpCSNGzdOu3fvtk579+7VoUOH8hRCclK8eHGbeYvFkm1bRkbGLe8jJw0aNJCvr68++OAD60fzt2rkyJH68ssvNWHCBP3www/avXu36tatm+ULStc/tsxgnV1b5uN94403NH36dL3wwgvasGGDdu/erfDw8By/+FRYHBwcshyj68ebZrr+Pyo5KV26tC5evJjrOlu3blWvXr3Uvn17ff3119q1a5dGjx6d6/GU8n7+XLhwQb6+vjetFUDhIbgCuG2BgYHatGmT4uLibMJrw4YNdfDgQVWpUiXLdP3VrUw1a9bUnj17dOXKFWvb5s2b5eDgoOrVq992ndu2bbP+nZaWpp07d6pmzZrWfW/dutUmaG3evFkeHh4qX768ta1y5crasGGDli9frqFDh95WPZs3b1ZkZKS6du2qunXryt/fX8eOHbutPjP77dy5s3r37q169eqpUqVK+uOPP2673+vVrFkzyxjQ64+vJPn6+lrvUpDpVm8N1qBBAx04cCDX/yxs2bJFgYGBGj16tO6++25VrVpVx48fv6X9ZWffvn1q0KBBgfUHIP8IrgAKREBAgDZu3KizZ88qPDxciYmJeuWVV7Rw4UKNGzdO+/fv1++//67FixdrzJgx2fbRq1cvubi4qF+/ftq3b582bNigoUOHqk+fPtaPw29HdHS0vvzySx04cECDBw/WxYsXNWDAAEnS008/rRMnTmjo0KE6cOCAli9frrFjx2rEiBFZQna1atW0YcMGffHFF3n6QYKDBw/aXHXevXu3rl27pqpVq2rp0qXavXu39uzZo0cffbRArhJXrVpVa9as0ZYtW/T7779r0KBBio+Pv+1+r/fMM89o1apVevPNN3Xo0CHNmDEjyzCB1q1b6+eff9bChQt16NAhjR07Vvv27bul/d13331KSkrK9XZUVatWVWxsrBYvXqzDhw/rnXfe0ZdffnlL+8vODz/8oLZt2xZYfwDyj+AKoMCUL19eGzdu1Pnz5xUeHq7Q0FB9/fXX+u6779SoUSM1adJE06ZNU2BgYLbblyhRQqtXr9aFCxfUqFEjPfTQQ2rTpo1mzJhRIPVNmjRJkyZNUr169fTjjz/qq6++UunSpSX9M+Thm2++0U8//aR69erpySef1MCBA3MM2dWrV9f69ev1ySef3PSetT169FCDBg1spvj4eE2dOlXe3t5q2rSpOnXqpPDwcDVs2PC2H+eYMWPUsGFDhYeHq1WrVvL3989yW6rb1aRJE7333nuaPn266tWrp++++y7LsQoPD9fLL7+s559/Xo0aNdLly5fVt2/fW9pfqVKl1LVr11xvPfbggw8qKipKQ4YMUf369bVlyxa9/PLLt7S/G23dulWXLl3SQw89VCD9Abg1FuN2B2kBQBF37NgxBQUFadeuXapfv769y8Et+vXXX3X//ffr8OHDcnd3V0pKilxcXLRmzRqFhYUV6r4feeQR1atXTy+99FKh7gdA7rjiCgAwheDgYE2ePFlHjx5VYmKiPvnkEzk4OKhGjRqFut/U1FTVrVtXUVFRhbofADfH7bAAAKYRGRkpSYqKitLHH3+syZMn23x5rjA4OTnlOGQEwJ3FUAEAAACYAkMFAAAAYAoEVwAAAJgCwRUAAACmQHAFAACAKRBcAQAAYAoEVwAAAJgCwRUAAACmQHAFAACAKfw/rUn8UFOiPHAAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Insight:**"
      ],
      "metadata": {
        "id": "U_253PkN8wMr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "diet = df.groupby('diet_quality_encoded')[['exam_score']].mean()\n",
        "print(diet)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WRDQtTZC8x1U",
        "outputId": "965c5971-e040-4083-caae-13ff12c81df4"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                      exam_score\n",
            "diet_quality_encoded            \n",
            "0                      70.240603\n",
            "1                      69.416845\n",
            "2                      67.586188\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "diet = df.groupby('diet_quality_encoded')['exam_score'].mean()\n",
        "plt.pie(diet, labels=diet.index, autopct='%1.1f%%')\n",
        "plt.title('Rata-rata Nilai Siswa berdasarkan Diet')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "id": "lYjb8TEl9GXF",
        "outputId": "f0486b24-7a2f-4bb3-a53c-5b86f3fb5a36"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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a1f9U/INN9zAlClTGD9+PL1792bnzp18+umnNW7HqK3XXnuNvLw8HnroIfz9/RkzZgy9e/cmODiYsWPH8pe//AWDwcDcuXPrZTioJqdOneK///0vULF2sGfPHubPn09qaip//etfmThx4nlv/69//Yuff/6Znj17ct9999G+fXuys7P57bffWLFiBdnZ2QDEx8cTFBTEzJkz8ff3x9fXl549e55320ujavT9nc7jz7upnc1ms2nx8fFafHy8Vl5ertntdu2f//yn1qJFC81sNmvdunXTli1bpo0dO1Zr0aJFldtu2LBB6969u+bl5VVlN7KTJ09qw4cP14KCgrTAwEBt5MiRWkpKyjl3YT3bn3f7PNuePXu0q6++WvPz89PCwsK0++67T9u+fXu1XdHKy8u1Rx55RAsPD9cMBkOVXUA/+ugjrVWrVprZbNbatm2rJSUl1bibaF2ylZWVafHx8Re9S+rSpUu1zp07a97e3lrLli21qVOnah9//HGV3UU1rXa7pKalpWn/+te/tMTERK1Jkyaah4eHFhwcrA0aNEhbsGBBleuevUvqb7/9pt1+++1a8+bNNbPZrEVERGg33HCDtnXr1mr3M3LkSA3QvvjiC/2y0tJSzcfHR/Py8tKKi4urXD8nJ0cbP368FhYWpvn5+WmDBw/WkpOTtRYtWmhjx44972OqfB5fe+017Y033tBiYmI0s9ms9evXT9u+fXu16x86dEi7++67taioKM3T01OLjo7WbrjhhiqP/3yvC03TtHfffVeLjY3VzGazdvnll2tr1qyp9vzX9vWyYMEC7dprr9UiIiI0Ly8vrXnz5trEiRO106dP69exWq3aX//6V61JkyaaxWLR+vTpo23cuLHafVbukjp//vxqmWt6TDabTbv99ts1Dw8P7auvvtI0TdPWr1+v9erVS7NYLFrTpk21v/3tb9r333+vAdrPP/+s3zYxMVHr0KFDtfup6f2gJi1atNAADdAMBoMWEBCgdejQQbvvvvu0zZs313ibmt4n0tLStIceekiLiYnRPD09taioKO2qq67SZs2aVeV6S5Ys0dq3b695eHg43O6pBk1roNoVQgjhdJxim4IQQojGIaUghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBCJ6UghBBC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3N3+bvpE91GWISMjg4iICFavXk3//v2V5RDnJqUgasVmt/HziZ9ZfHAxG05tkIPnOYlwSzg3xN3AzQk3Ex8UrzoOBw8epFWrVuzcuZOOHTuqjiNqIKUg6iyrOIvlR5bz9eGv2ZO1R3UccRZvkzcDmw/k5vib6dWkFyajSXUkAOx2OzfddBO5ubmsW7dOdRxxDlIK4pIcyj3E14e+5psj35BamKo6jtsyYKBbRDduTriZa1tci5+Xn+pI1Tz44IMsX76cdevW0axZM9VxxDlIKYh6YdfsbEndwrLDy1hzcg3Z1mzVkVyeAQMdQjswIGYAQ+OG0szfcd9oH374YZYsWcKaNWuIjZXdXh2ZlIKod3bNzq7MXaw+uZq1J9eyN3uv6kguw+Jh4comVzIgZgD9mvVz+BnHmqbxyCOPsHjxYlatWkWrVq1URxIXIKUgGlx6UTprT65lzck1bDq9SXZxraMmvk1IbJZIYkwiPaJ64GXyUh2p1iZNmsRnn33GkiVLqsxNCAwMxGKxKEwmzkVKQTSqUlspW9O28mvar2zP2M7OjJ1SEmdp5teMbhHd6BrRlcsiLiMhOEF1pItmMBhqvDwpKYlx48Y1bhhRK1IKQimb3caB3ANsS9/GtoxtbEvf5lbzITyMHrQPaU/XiK50jehKt4huDj8kJFyblIJwOJnFmWxP386+nH0czjvM4bzDHMs7Rqm9VHW0SxLiHUJ8UDxxgXHEB8XTOrg1HUI74O3hrTqaEDopBeEU7JqdU/mn9JKo/DqZf5Icaw4ajvFnbPGwEGYJI9ovukoBxAfGE+QdpDqeEBckpSCcXrm9nKziLDKtmRX/FmeSUZRBZnEmWdYssoqzKC4vptRWSomtpOJfe8W/pbZSbJqt2jJNBhNeJi/MJjM+Hj5YPCz4ePrg4+FDiCWEcEs4ET4RhFnCCLeEE+YTRoQlwiHnBwhRF1IKwu2V28sptZVi1+x4mjzxNHpiNBhVxxJCCSkFIYQQOvk4JIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQielIIQQQvf/+FpyUGe5YJ0AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Hitung matriks korelasi\n",
        "correlation_matrix = df.corr()\n",
        "\n",
        "# Buat heatmap\n",
        "plt.figure(figsize=(10, 8)) # Atur ukuran figure\n",
        "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=\".2f\")\n",
        "plt.title('Heatmap Korelasi Antar Fitur')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 590
        },
        "id": "sQ1wnWzTmoPW",
        "outputId": "a12ae104-a47c-4732-82b2-29ebbf596766"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x800 with 2 Axes>"
            ],
            "image/png": "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7zKrV1tqmP37mxX1jY737tygKn0voPIh0Ov11ecbnKq0PXv2YOzYsUhNTYVMZnq667I4auH93Pb1X7Jz6onKzkKVk5VefSIsFcnexf6fE5GR9GTTXSepZOmJqZWdhSrHxtb8u23JvLzc/9bDp+fht22NKjsLZt3s2anSjt3izPl/TlTJGKGkSrN7927UqVMHNWvWxO3btzFnzhwMGzbsuTYmiYiIiIio/LBBSZUmNjYWCxYsQGxsLFxcXDB06FAsX768srNFRERERGQg5GtDSsQGJVWa2bNnY/bs2ZWdDSIiIiIiKqMXa2YTIiIiIiIiqjIYoSQiIiIiIjJD8F99X0opMUJJREREREREZcIIJRERERERkRmCF+z95y8alg4RERERERGVCRuUREREREREVCbs8kpERERERGQGJ+UpGSOUREREREREVCaMUBIREREREZkhFDFCWRJGKImIiIiIiKhM2KAkIiIiIiKiMmGXVyIiIiIiIjM4KU/JGKEkIiIiIiKiMmGEkoiIiIiIyAyBkDG4krB0iIiIiIiIqEwYoSQiIiIiIjKDYyhLxgglERERERERlQkblERERERERFQm7PJKRERERERkBru8lowRSiIiIiIiIioTRiiJiIiIiIjMYISyZIxQEhERERERUZmwQUlERERERERlwi6vREREREREZgiEjMGVhKVDREREREREZcIIJRERERERkRlCESflKQkjlERERERERFQmjFASERERERGZwdeGlIwRSiIiIiIiIioTNiiJiIiIiIioTNjllaq9nVNPVHYWqqQxG/pWdhaqnN0f/l7ZWaiSEqISKjsLVU6uJqeys1AlyRTWlZ2FKmeL9ZLKzkKV9MbD8ZWdBXqO+NqQkrF0iIiIiIiIqEwYoSQiIiIiIjKDk/KUjBFKIiIiIiIiKhM2KImIiIiIiKhM2OWViIiIiIjIDHZ5LRkjlERERERERFQmjFASERERERGZwdeGlIylQ0RERERERGXCCCUREREREZEZHENZMkYoiYiIiIiIqEzYoCQiIiIiIqIyYZdXIiIiIiIiMzgpT8lYOkRERERERFQmjFASERERERGZI+CkPCVhhJKIiIiIiIjKhA1KIiIiIiIiKhN2eSUiIiIiIjKD76EsGSOUREREREREVCaMUBIREREREZnB14aUjKVDREREREREZcIIJRERERER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EREREdFzJBAIK22pCv6TXV5NiY+PR1qa6W5XCoUCTk5OFZyj0omKikJ2tukZ4Ozs7GBnZ1cux61K5cUur2XDLq/Pjl1ey4ZdXp8du7yWDbu8Pjt2eS0bdnl9di9yl9fk5ZMq7di2n7z471P/T3V5LYmTk9ML1QgqrZo1a1bKcatqeRERERER0fPDBiUREREREZE51eU9UOWkanTMJSIiIiIiohcOI5RERERERERmCISMwZWEpUNERERERERlwgYlERERERERlQm7vBIREREREZkh4KQ8JWKEkoiIiIiIiMqEEUoiIiIiIiJzBIzBlYSlQ0RERERERGXCBiURERERERGVCbu8EhERERERmcFJeUrGCCURERERERGVCSOURERERERE5ggZgysJS4eIiIiIiIjKhBFKIiIiIiIiMwQCjqEsCSOUREREREREVCZsUBIREREREVGZsMsrERERERGROZyUp0QsHSIiIiIiIioTRiiJiIiIiIjMEAg5KU9JGKEkIiIiIiKiMmGDkoiIiIiIiMqEXV6JiIiIiIjMETAGVxKWDhEREREREZUJI5RERERERETmcFKeEjFCSURERERERGXCCCUREREREZEZAo6hLBFLh4iIiIiIiMqEDUoiIiIiIiIqE3Z5pWovKz2zsrNQJe3+8PfKzkKVM2rdy5WdhSppx+TjlZ2FKkdjwdt3Wei02srOQpWzzGVjZWehSnLIjKvsLNDzxEl5SsQIJREREREREZUJH3ESERERERGZIRAyBlcSlg4RERERERGVCRuUREREREREVCbs8kpERERERGSOgJPylIQRSiIiIiIiIioTRiiJiIiIiIjM4aQ8JWLpEBERERERUZkwQklERERERGQOx1CWiBFKIiIiIiIiKhM2KImIiIiIiKhM2OWViIiIiIjIDAEn5SkRS4eIiIiIiIjKhBFKIiIiIiIicwSMwZWEpUNERERERERlwgYlERERERERlQm7vBIREREREZkj5HsoS8IIJREREREREZUJI5RERERERERmCDgpT4lYOkRERERERFQmjFASERERERGZwzGUJWKEkoiIiIiIiMqEDUoiIiIiIiIqE3Z5JSIiIiIiMoeT8pSIpUNERERERERlwgglERERERGROQJOylMSRiiJiIiIiIioTNigJCIiIiIiojJhl1ciIiIiIiJzhIzBlYSlQ0RE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27c3uU3Hjh3x6NEj6HQ6w7rAwEC4uLiUujEJsEFJRERERERknkBYecszmDlzJrZt24Zdu3bB398fkyZNQmZmpmHW11GjRmHevHmG9JMmTUJSUhI+/PBDBAYG4ujRo1ixYgWmTJnyTMdll1ciIiIiIqIq7s0338STJ0+wYMECxMbGolmzZjhx4oRhop6IiAgIhYWNVDc3N5w8eRIzZsxA06ZNUbNmTXz44YeYM2fOMx2XDUoiIiIiIqJqYOrUqZg6darJz3x9fYuta9++PS5fvvyvjskGJRERERERkRn6Z+x6+l/D0iEiIiIiIqIyYYSSiIiIiIjInEp8bUhVwAglERERERERlQkjlERERERERGZwDGXJWDpERERERERUJmxQEhERERERUZmwyysREREREZE5nJSnRIxQEhERERERUZkwQklERERERGQOJ+UpEUuHiIiIiIiIyoQNSiIiIiIiIioTdnklIiIiIiIyQ89JeUrECCURERERERGVCSOURERERERE5nBSnhKxdIiIiIiIiKhMGKEkIiIiIiIyQw+OoSwJI5RERERERERUJmxQEhERERERUZmwyysREREREZEZek7KUyKWDhEREREREZUJI5RERERERETmMEJZIpYOERERERERlQkblERERERERFQm7PJKRERERERkhl7A91CWhBFKIiIiIiIiKhNGKImIiIiIiMzga0NKxtJ5wYwZMwaDBg2q7GwY8fDwwNq1ays7G0RERERE9IJhhJLoOerZxgqvdLKG0kaEyNhcfHc0DSFRuWbTt35Jitd7yuGgEiEuKQ8/nUzHnSCNybRjBirQo4019hxLxclLWeV1CpXi7UFO6NvFDtZWIjx4lIWNu6MQHZ9T4jYDetjh9b6OsFWKERqpxuY90QgMzQYA2FiL8PZrTmjRWA5HOwukpufh0q00fHcoDlnZuoo4pXLVt5MCr/ZQQqUQITwqB98eTMSjCNP1BgDaN7PG8Fds4WgnRsyTPHz/ayJuPcg2fN62qRVe7qhAHTcJ5NYizPrfY4RFlVz+VY1dp1ao89F7ULZoDKmrE66/PhlxR86UvE2XNmi0ei5sGtWHOjIGj1ZuxuPdh4zSuE8aiToz34OkhiPS7jzE/elLkXrtbnmeSoXq01GeX9fkIoRH52D7z4l4FGG+brTzscLwfvl1LfZJHr7/LQm3/AvrWpsm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86AJFKVeHuON/tiRR4CfkREDggkKMBF33Mzz76bYu+WHBbtx5kiGQ3FmXv8klZHXBTPq+r/OtfkpJCSfca619OH+UWXvtHt4TOl5t2h5BotXZF6Scl0K3/6SjpeHgftGReLrY2R/TAEz/5PocF+LDPcg4Izuiuu25RDgZ2LUoDCCA0zEJpqZ+Z9Eh+6IHy5OxWYL5/G7auHuZmD7vjze+dL1x4bvOmrB1xuu7eSGv4+BE6dszP/ZbJ+oJ8jP4HCuBfgYmDS0LNLWo607Pdq6E3PCwns/lfby6dqy9BH3nsGOPYEWrS19ncjlYMPOfAL8MhjeP4ggfxNxJ4qY/UGq/RoNPev34NAxM//5LI0RA4K4dWAwyWnFvLIw1X6NhgSa6Nyq9Lfz5UdqOuxr5rxk9sVc3r0NpPIMtot5GaG4pJ49e9KuXTvmzp1b3VmpFqOfSTp/IjlH2nHXHwd2qfn4+54/kZxj3NsDqzsLLmfhxOXVnQWXVFSoSVkuVrP2das7Cy4p7lBqdWfB5Xz1asUz+Venk/s2V9u+w1t0qbZ9X6h/XphHREREREREXIK6vP6P+/333xk40Hl04PRMsyIiIiIi/4tsisFVSBXK/wFr1651+l2nTp3YsWPHJcuLiIiIiIhcPlSh/B/n7e1No0aNqjsbIiIiIiLiglShFBERERERccJmcPF3jlUxdQgWERERERGRSlGEUkRERERExAmbQTG4iujoiIiIiIiISKWoQikiIiIiIiKVoi6vIiIiIiIiTtjQpDwVUYRSREREREREKkURShERERERESc0KU/FdHRERERERESkUlShFBERERERkUpRl1cREREREREnbAZNylMRRShFRERERESkUhShFBERERERcUKvDamYIpQiIiIiIiJSKYpQioiIiIiIOKHXhlRMR0dEREREREQqRRVKERERERERqRR1eRUREREREXFCk/JUTBFKERERERERqRRFKEVERERERJzQpDwV09ERERERERGRSlGFUkRERERERCpFXV5FRERERESc0KQ8FVOEUkRERERERCpFEUoREREREREnNClPxXR0REREREREpFIUoRQREREREXFCYygrpgiliIiIiIiIVIoqlCIiIiIiIlIp6vIql72cjNzqzoJLKjYXVXcWXI7ZXbfUylg4cXl1Z8HljH1zQHVnwSW9N+GH6s6Cy/H1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Yp999hl+fn7lflq2bFnd2bsorlAWg8HAzz//zDXXXEPHjh358ccf+eabb+jbt291Z01ERERE/kdoDGXFFKG8CDfccANXXHFFud+d/XqOfzpXKIu3tzerVq2q7myIiIiIiIgTqlBeBH9/f/z9/c+f0AVcTmUREREREZHqoQqliIiIiIiIE67S9bS6aAyliIiIiIiIVIoilCIiIiIiIk5YFaGskCKUIiIiIiIiUimqUIqIiIiIiEilqMuriIiIiIiIE5qUp2KKUIqIiIiIiEilKEIpIiIiIiLihNVa3Tn4Z1OEUkRERERERCpFFUoRERERERGpFHV5FRERERERcUKT8lRMEUoRERERERGpFEUoRUREREREnFCEsmKKUIqIiIiIiEilKEIpIiIiIiLihFURygopQikiIiIiIiKVogqliIiIiIiIVIq6vIqIiIiIiDhhq9ZZeQzVuO8LowiliIiIiIiIVIoilCIiIiIiIk7otSEVU4RSREREREREKkUVShERERERkcvAW2+9RXR0NF5eXlxxxRVs3rzZadr333+fq6++muDgYIKDg+nbt2+F6Z1RhVJERERERMQJq7X6Phfjq6++YvLkyUyfPp0///yTtm3b0r9/f1JTU8tNv3btWkaOHMmvv/7Khg0bqFOnDtdeey3Hjx+/qP2qQikiIiIiIuLi5syZw5133sm4ceNo0aIF77zzDj4+PsyfP7/c9J999hn33Xcf7dq1o1mzZnzwwQdYrVZWr159UfvVpDwiIiIiIiJOVOekPGazGbPZ7LDM09MTT09Ph2VFRUVs27aNqVOn2pcZjUb69u3Lhg0bLmhf+fn5FBcXExISclF5VIRSRERERETkH2j27NkEBgY6fGbPnn1OurS0NCwWCxEREQ7LIyIiSE5OvqB9Pf7449SsWZO+ffteVB4VoRQREREREXHCWo0RyqlTpzJ58mSHZWdHJ/8OL774Il9++SVr167Fy8vrotZVhVJEREREROQfqLzureUJCwvDZDKRkpLisDwlJYXIyMgK13311Vd58cUXWbVqFW3atLnoPKrLq4iIiIiIiAvz8PCgY8eODhPqnJ5gp2vXrk7Xe/nll5k1axbLly+nU6dOldq3IpRy2fMLDqjuLLikd3yfre4suJznot6q7iy4JF8/j+rOgsvx9XWv7iy4JONde6o7Cy7H2qVVdWfBJcXcuLC6s+CCwqs7A05V56Q8F2Py5MmMGTOGTp060aVLF+bOnUteXh7jxo0DYPTo0dSqVcs+BvOll15i2rRpfP7550RHR9vHWvr5+eHn53fB+1WFUkRERERExMWNGDGCkydPMm3aNJKTk2nXrh3Lly+3T9QTHx+P0VjWQXXevHkUFRVxyy23OGxn+vTpzJgx44L3qwqliIiIiIiIE7bqnJUHw0WlnjhxIhMnTiz3u7Vr1zr8HRcXV8k8OdIYShEREREREakUVShFRERERESkUtTlVURERERExIlq7fHqAhShFBERERERkUpRhFJERERERMQJV3ltSHVRhFJEREREREQqRRFKERERERERJ6waRFkhRShFRERERESkUlShFBERERERkUpRl1cREREREREnNClPxRShFBERERERkUpRhFJERERERMQJRSgrpgiliIiIiIiIVIoqlCIiIiIiIlIp6vIqIiIiIiLihFV9XiukCKWIiIiIiIhUiiKUIiIiIiIiTtis1Z2DfzZFKEVERERERKRSFKEUERERERFxwqYxlBVShFJEREREREQqRRVKERERERERqRR1eRUREREREXHCqkl5KqQIpYiIiIiIiFSKIpQiIiIiIiJOaFKeiilCKSIiIiIiIpWiCqWIiIiIiIhUirq8ioiIiIiIOGFVj9cKKUIpIiIiIiIilaIIpYiIiIiIiBM2hSgrpAiliIiIiIiIVIoilCJ/s5GDQul3VSC+3kYOHC3gnS9SSTpZXOE6A68J5KZ+IQQFmIhLNPP+opMcPlZo/97dzcC4oeF07+iPu5uBHfvzeOfLVLJyLFVdnCrn060vvj2ux+gfSHFSPDlLPqY44ajz9N3749O1L6bgUKx5ORTu2kzOskVQUnqMfXsNxqt1Z0zhUdhKiiiOO0zOz19hOZl0qYpU5a5p60GfTp4E+Bo4ftLC4l8LOZbs/Fxo39iN66/yIjTAyMlMK0t+L2RfbIn9e38fAzde7UXzem54exo4cryExWsKOZl5eb3JuWsLE9e0dcPf20BSuo3v1xeReLL8VueIYAP9OrlTK8xAiL+RH/8oYt0ex2Pcs50braJN1AgyUGyBYylWft5UTFrW5dOS3amJgW7Njfh5Q0oGLNtq4cSp8tOGB0LPNkaiQgwE+RlYsdXCpoPnHgt/b+jT3kijmgbcTZCeCz9ssJCUXsWFuYR+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pFxevw3y9zDYNMpULnND7iu/K65U7B3fZ6s7Cy7nuai3qjsLLsnXT2NyLpavr+tHpapD5xZ65LlY1i6tzp9IzvHKjQurOwsuZ+3XXc+fqJpMmZd//kRV5NV7fc6fqJppDKWIiIiIiIhUiiqUAsALL7yAn59fuZ+BAwdWd/ZERERERKqFzWarto8r0BhKAeCee+5h+PDh5X73d70WRERERERELi+qUAoAISEhhISEVHc2RERERETEhahCKSIiIiIi4oTV6hpdT6uLxlCKiIiIiIhIpShCKSIiIiIi4oSLzI1TbRShFBERERERkUpRhVJEREREREQqRV1eRUREREREnLBpUp4KKUIpIiIiIiIilaIIpYiIiIiIiBNWzcpTIUUoRUREREREpFJUoRQREREREZFKUZdXERERERERJzQpT8UUoRQREREREZFKUYRSRERERETECUUoK6YIpYiIiIiIiFSKIpQiIiIiIiJOKEBZMUUoRUREREREpFJUoRQREREREZFKUZdXERERERERJzQpT8UUoRQREREREZFKUYRSRERERETECZtNEcqKKEIpIiIiIiIilaIKpYiIiIiIiFSKuryKiIiIiIg4YdWkPBVShFJEREREREQqRRFKERERERERJzQpT8UUoRQREREREZFKUYRSRERERETECZvGUFZIEUoRERERERGpFFUoRUREREREpFLU5VUueyXFJdWdBZd0y4E7qzsLLicsL6W6s+CSPH08qzsLLsfT26O6s+CSjh3VY8/FirlxYXVnwSU9+v3Y6s6CCzpY3RlwSl1eK6YIpYiIiIiIiFSKmupEREREREScsOq1IRVShFJEREREREQqRRVKERERERERqRR1eRUREREREXFCk/JUTBFKERERERERqRRFKEVERERERJywaVKeCilCKSIiIiIiIpWiCKWIiIiIiIgTVo2hrJAilCIiIiIiIlIpqlCKiIiIiIhIpajLq4iIiIiIiBN6bUjFFKEUERERERGRSlGEUkRERERExAm9NqRiilCKiIiIiIhIpahCKSIiIiIiIpWiLq8iIiIiIiJO2KzW6s7CP5oilCIiIiIiIlIpilCKiIiIiIg4YdVrQyqkCKWIiIiIiIhUiiKUIiIiIiIiTui1IRVThFJEREREREQqRRVKERERERERqRR1eRUREREREXHCpkl5KqQIpYiIiIiI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Vfw+uPQQs5ZdVP59YKENo2e2iRyJzUFdQy2MGmwFk7paSM0owKzlF3H/mUY9Yb9mo7aWBiaOtIFBnVpIu1aIGUsv4Nbtmv1zatTWEZ2idyhet/h8LgDgesg+JI35BNoNzKBr9ed2p8cZN3DawxctAj+BzWQfFN3IxjnfebgXdUJxT9b3B1HbzARNFvpBu74ZHiZexKl+Y1F8Rz0extK/QyZ/kX60ak4mk+GHH36Ap6en6CgAgKVLl+L7779HUlKS6Cg1Wr9xF0RHqJHysmte0x/R6j3zi5v+OW29qo/ioMpp675Ze79eFa3akm8dITlpiamiI9RIM38cKTpCjfOfksuiI1TpP2OThf27f938Yg1yRFCDY9HVT35+PpKTkxEcHIzJkyeLjkNERERERFQptSwod+3ahTp16lT61bJlS9Hx/takSZPQtm1bdOvWTWW56/jx46v8b/vrERtERERERPRyuIfy+dRy7YeHh4fSOZfP+utxGc+Syurfbdu2Ydu2bZVeW7JkiaL5zV8ZGhr+i6mIiIiIiIiUqWVBaWBgAAMDg7+/sQYyNzeHubm56BhERERERETqWVASERERERG9CjVl6akoarmHkoiIiIiIiP59nKEkIiIiIiKqQvmzB+6SCs5QEhERERERUbWwoCQiIiIiIqJq4ZJXIiIiIiKiKrApz/NxhpKIiIiIiIiqhTOUREREREREVZCXsynP83CGkoiIiIiIiKqFBSURERERERFVC5e8EhERERERVYFNeZ6PM5RERERERERULZyhJCIiIiIiqoJczqY8z8MZSiIiIiIiIqoWzlASERERERFVoZx7KJ+LM5RERERERERULSwoiYiIiIiIqFq45JWIiIiIiKgK8nI25XkezlASERERERFRtXCGkoiIiIiIqApyNuV5Ls5QEhERERERUbWwoCQiIiIiIqJq4ZJXIiIiIiKiKsjlbMrzPJyhJCIiIiIiomrhDCUREREREVEV2JTn+ThDSURERERERNXCGUoiIiIiIqIqyMu5h/J5OENJRERERERE1cKCkoiIiIiIiKpFJpfLucuUSIAnT55g5cqV+OSTT6CtrS06To3A96x6+L69OL5n1cP37cXxPasevm8vju8Z/VtYUBIJ8vDhQxgZGeHBgwcwNDQUHadG4HtWPXzfXhzfs+rh+/bi+J5VD9+3F8f3jP4tXPJK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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Insight:**"
      ],
      "metadata": {
        "id": "92L_9nHD9oEc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Data Splitting**"
      ],
      "metadata": {
        "id": "psPvjIK09zaj"
      }
    },
    {
      "source": [
        "# Variabel independen (fitur)\n",
        "X = df[['mental_health_rating', 'study_hours_per_day', 'exercise_frequency', 'sleep_hours']]\n",
        "\n",
        "# Variabel dependen\n",
        "y = df['exam_score']\n",
        "\n",
        "# Data splitting\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "cell_type": "code",
      "metadata": {
        "id": "W50HwzrW_bEE"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)"
      ],
      "metadata": {
        "id": "e5-zblJNDkz_"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Modeling**"
      ],
      "metadata": {
        "id": "t8k4kqf3_ngm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Random Forest**"
      ],
      "metadata": {
        "id": "Mf10GMbM_vGC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)\n",
        "rf.fit(X_train, y_train)\n",
        "y_pred = rf.predict(X_test)"
      ],
      "metadata": {
        "id": "9DdnFKsl_qWj"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Decision Tree**"
      ],
      "metadata": {
        "id": "V42MQZNqBxCz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\n",
        "\n",
        "# Inisialisasi dan latih model\n",
        "dt = DecisionTreeRegressor(max_depth=10, random_state=42)\n",
        "dt.fit(X_train, y_train)\n",
        "\n",
        "# Prediksi\n",
        "y_pred_dt = dt.predict(X_test)"
      ],
      "metadata": {
        "id": "LE8xkpgWBwSo"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Linear Regression**"
      ],
      "metadata": {
        "id": "YUfcMRj1CDx5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Inisialisasi dan latih model\n",
        "lr = LinearRegression(n_jobs=-1, positive=True)\n",
        "lr.fit(X_train, y_train)\n",
        "\n",
        "# Prediksi\n",
        "y_pred_lr = lr.predict(X_test)"
      ],
      "metadata": {
        "id": "-os4R4OdCJE4"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Evaluation**"
      ],
      "metadata": {
        "id": "vVBK0VbXCTcv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_model(y_true, y_pred, model_name):\n",
        "    print(f\"\\nEvaluasi Model {model_name}:\")\n",
        "    print(f\"MAE: {mean_absolute_error(y_true, y_pred):.4f}\")\n",
        "    print(f\"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}\")\n",
        "    print(f\"R2 Score: {r2_score(y_true, y_pred):.4f}\")\n",
        "\n"
      ],
      "metadata": {
        "id": "U3W_QVAsiPnY"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "source": [
        "evaluate_model(y_test, y_pred, \"Random Forest\")\n",
        "evaluate_model(y_test, y_pred_dt, \"Decision Tree\")\n",
        "evaluate_model(y_test, y_pred_lr, \"Linear Regression\")"
      ],
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LEAX_Gy7jM7J",
        "outputId": "d1c35e63-6e9d-47c8-d36f-9bf806cffa8d"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Evaluasi Model Random Forest:\n",
            "MAE: 5.7304\n",
            "RMSE: 7.0598\n",
            "R2 Score: 0.8074\n",
            "\n",
            "Evaluasi Model Decision Tree:\n",
            "MAE: 7.3959\n",
            "RMSE: 9.4541\n",
            "R2 Score: 0.6546\n",
            "\n",
            "Evaluasi Model Linear Regression:\n",
            "MAE: 5.3687\n",
            "RMSE: 6.6054\n",
            "R2 Score: 0.8314\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Hyperparameter Tuning**"
      ],
      "metadata": {
        "id": "v0DgZG5dlGn3"
      }
    },
    {
      "source": [
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "param_grid = {\n",
        "    'fit_intercept': [True, False],\n",
        "    'positive': [True, False]\n",
        "}\n",
        "\n",
        "grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='r2')\n",
        "grid_search.fit(X_train, y_train)\n",
        "\n",
        "best_lr = grid_search.best_estimator_"
      ],
      "cell_type": "code",
      "metadata": {
        "id": "_CUo-F6spq9k"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "source": [
        "y_pred_tuned = best_lr.predict(X_test)"
      ],
      "cell_type": "code",
      "metadata": {
        "id": "Y4s7jFgqlr6-"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "source": [
        "# Evaluasi model tuned\n",
        "evaluate_model(y_test, y_pred_tuned, \"Random Forest (Tuned)\")"
      ],
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ToJHdPGTluNV",
        "outputId": "97ace31c-8a34-4836-f5ef-b79f58db7c6f"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Evaluasi Model Random Forest (Tuned):\n",
            "MAE: 5.3687\n",
            "RMSE: 6.6054\n",
            "R2 Score: 0.8314\n"
          ]
        }
      ]
    }
  ]
}