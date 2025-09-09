{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOz9o/rT6ftm9H8+aWMyw48",
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
        "<a href=\"https://colab.research.google.com/github/Alphaz-006/projects/blob/main/healthcare_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install opendatasets xgboost graphviz lightgbm scikit-learn"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "dS1V4jBPlGFk",
        "outputId": "5a12245f-c8ae-427c-87a2-2306f6d6d4e6"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting opendatasets\n",
            "  Downloading opendatasets-0.1.22-py3-none-any.whl.metadata (9.2 kB)\n",
            "Requirement already satisfied: xgboost in /usr/local/lib/python3.12/dist-packages (3.0.4)\n",
            "Requirement already satisfied: graphviz in /usr/local/lib/python3.12/dist-packages (0.21)\n",
            "Requirement already satisfied: lightgbm in /usr/local/lib/python3.12/dist-packages (4.6.0)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from opendatasets) (4.67.1)\n",
            "Requirement already satisfied: kaggle in /usr/local/lib/python3.12/dist-packages (from opendatasets) (1.7.4.5)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.12/dist-packages (from opendatasets) (8.2.1)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from xgboost) (2.0.2)\n",
            "Requirement already satisfied: nvidia-nccl-cu12 in /usr/local/lib/python3.12/dist-packages (from xgboost) (2.27.3)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from xgboost) (1.16.1)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)\n",
            "Requirement already satisfied: bleach in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (6.2.0)\n",
            "Requirement already satisfied: certifi>=14.05.14 in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (2025.8.3)\n",
            "Requirement already satisfied: charset-normalizer in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (3.4.3)\n",
            "Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (3.10)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (5.29.5)\n",
            "Requirement already satisfied: python-dateutil>=2.5.3 in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (2.9.0.post0)\n",
            "Requirement already satisfied: python-slugify in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (8.0.4)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (2.32.4)\n",
            "Requirement already satisfied: setuptools>=21.0.0 in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (75.2.0)\n",
            "Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (1.17.0)\n",
            "Requirement already satisfied: text-unidecode in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (1.3)\n",
            "Requirement already satisfied: urllib3>=1.15.1 in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (2.5.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.12/dist-packages (from kaggle->opendatasets) (0.5.1)\n",
            "Downloading opendatasets-0.1.22-py3-none-any.whl (15 kB)\n",
            "Installing collected packages: opendatasets\n",
            "Successfully installed opendatasets-0.1.22\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import opendatasets as od\n",
        "import pandas as pd"
      ],
      "metadata": {
        "id": "I9JO1YAalNBo"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler"
      ],
      "metadata": {
        "id": "btoBD4lrlNTa"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "od.download('https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection/data')# Download from Kaggle\n",
        "od.download('https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset')\n",
        "od.download('https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link')# Download from Kaggle"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iR5zX8ezoJHq",
        "outputId": "1beff3af-a984-4e9c-c2ab-2a10d1dac1b2"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds\n",
            "Your Kaggle username: gaarneesh\n",
            "Your Kaggle Key: ··········\n",
            "Dataset URL: https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection\n",
            "Downloading parkinson-disease-detection.zip to ./parkinson-disease-detection\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 14.8k/14.8k [00:00<00:00, 21.8MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds\n",
            "Your Kaggle username:"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " gaarneesh\n",
            "Your Kaggle Key: ··········\n",
            "Dataset URL: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset\n",
            "Downloading diabetes-prediction-dataset.zip to ./diabetes-prediction-dataset\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 734k/734k [00:00<00:00, 614MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds\n",
            "Your Kaggle username:"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " gaarneesh\n",
            "Your Kaggle Key: ··········\n",
            "Dataset URL: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset\n",
            "Downloading heart-disease-dataset.zip to ./heart-disease-dataset\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 6.18k/6.18k [00:00<00:00, 9.35MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Skipping, found downloaded files in \"./cancer-patients-and-air-pollution-a-new-link\" (use force=True to force download)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "od.download('https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IDY1F8_toxBg",
        "outputId": "b0541692-7590-4f9b-fd0d-eaf10d718ce9"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Skipping, found downloaded files in \"./heart-disease-dataset\" (use force=True to force download)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def load_diabetes():\n",
        "    # Example: using Pima Indians Diabetes Database\n",
        "\n",
        "    df = pd.read_csv('diabetes_prediction_dataset.csv')\n",
        "    X = df.drop('Outcome', axis=1)\n",
        "    y = df['Outcome']\n",
        "    return X, y"
      ],
      "metadata": {
        "id": "6TnQdHj8lNWz"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def load_heart():\n",
        "\n",
        "    df = pd.read_csv('heart.csv')\n",
        "    X = df.drop('target', axis=1)\n",
        "    y = df['target']\n",
        "    return X, y"
      ],
      "metadata": {
        "id": "xkcyKpBkle_6"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def load_cancer():\n",
        "    df = pd.read_csv('/content/cancer-patients-and-air-pollution-a-new-link/cancer patient data sets.csv')\n",
        "    # Assume binary classification for cancer diagnosis (modify as per dataset)\n",
        "    X = df.drop('diagnosis', axis=1)\n",
        "    y = LabelEncoder().fit_transform(df['diagnosis'])\n",
        "    return X, y"
      ],
      "metadata": {
        "id": "NdsA2ybilfL_"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def load_parkinsons():\n",
        "    df = pd.read_csv('parkinson-disease-detection/Parkinsson disease.csv')\n",
        "    X = df.drop(['status', 'name'], axis=1)  # 'status' is target, 'name' is ID\n",
        "    y = df['status']\n",
        "    return X, y"
      ],
      "metadata": {
        "id": "7ltsW7AOlfPI"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess(X):\n",
        "    scaler = StandardScaler()\n",
        "    X_scaled = scaler.fit_transform(X)\n",
        "    return X_scaled"
      ],
      "metadata": {
        "id": "uUkz02c4lfSU"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def train_and_evaluate(X, y):\n",
        "    X = preprocess(X)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(\n",
        "        X, y, test_size=0.2, random_state=42\n",
        "    )\n",
        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "    model.fit(X_train, y_train)\n",
        "    y_pred = model.predict(X_test)\n",
        "    print(classification_report(y_test, y_pred))\n",
        "    return model"
      ],
      "metadata": {
        "id": "GgIaJUD_lfV4"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def get_patient_input(X):\n",
        "    data = []\n",
        "    print(\"Enter the following details for prediction:\")\n",
        "    for feature in X.columns:\n",
        "        val = float(input(f\"{feature}: \"))\n",
        "        data.append(val)\n",
        "    return pd.DataFrame([data], columns=X.columns)\n"
      ],
      "metadata": {
        "id": "OfFKPZbNltUY"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        },
        "collapsed": true,
        "id": "XHHQBV2KlDlX",
        "outputId": "4ae9e7b4-3319-4c1b-e833-013eaff7a8ce"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Which disease do you want to predict?\n",
            "Options: diabetes, heart, cancer, parkinsons\n",
            "Enter your choice: cancer\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyError",
          "evalue": "\"['diagnosis'] not found in axis\"",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-2142713182.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     23\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"__main__\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 25\u001b[0;31m     \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/tmp/ipython-input-2142713182.py\u001b[0m in \u001b[0;36mmain\u001b[0;34m()\u001b[0m\n\u001b[1;32m      9\u001b[0m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_heart\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mchoice\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"cancer\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_cancer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mchoice\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"parkinsons\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_parkinsons\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/tmp/ipython-input-4294688769.py\u001b[0m in \u001b[0;36mload_cancer\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/cancer-patients-and-air-pollution-a-new-link/cancer patient data sets.csv'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0;31m# Assume binary classification for cancer diagnosis (modify as per dataset)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m     \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'diagnosis'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m     \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLabelEncoder\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'diagnosis'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, axis, index, columns, level, inplace, errors)\u001b[0m\n\u001b[1;32m   5579\u001b[0m                 \u001b[0mweight\u001b[0m  \u001b[0;36m1.0\u001b[0m     \u001b[0;36m0.8\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5580\u001b[0m         \"\"\"\n\u001b[0;32m-> 5581\u001b[0;31m         return super().drop(\n\u001b[0m\u001b[1;32m   5582\u001b[0m             \u001b[0mlabels\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5583\u001b[0m             \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, axis, index, columns, level, inplace, errors)\u001b[0m\n\u001b[1;32m   4786\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;32min\u001b[0m \u001b[0maxes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4787\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4788\u001b[0;31m                 \u001b[0mobj\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_drop_axis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   4789\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4790\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0minplace\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_drop_axis\u001b[0;34m(self, labels, axis, level, errors, only_slice)\u001b[0m\n\u001b[1;32m   4828\u001b[0m                 \u001b[0mnew_axis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4829\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4830\u001b[0;31m                 \u001b[0mnew_axis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   4831\u001b[0m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_indexer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnew_axis\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4832\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, errors)\u001b[0m\n\u001b[1;32m   7068\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmask\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   7069\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0merrors\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m\"ignore\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 7070\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"{labels[mask].tolist()} not found in axis\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   7071\u001b[0m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mindexer\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m~\u001b[0m\u001b[0mmask\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   7072\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdelete\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyError\u001b[0m: \"['diagnosis'] not found in axis\""
          ]
        }
      ],
      "source": [
        "def main():\n",
        "    print(\"Which disease do you want to predict?\")\n",
        "    print(\"Options: diabetes, heart, cancer, parkinsons\")\n",
        "    choice = input(\"Enter your choice: \").lower()\n",
        "\n",
        "    if choice == \"diabetes\":\n",
        "        X, y = load_diabetes()\n",
        "    elif choice == \"heart\":\n",
        "        X, y = load_heart()\n",
        "    elif choice == \"cancer\":\n",
        "        X, y = load_cancer()\n",
        "    elif choice == \"parkinsons\":\n",
        "        X, y = load_parkinsons()\n",
        "    else:\n",
        "        print(\"Invalid selection.\")\n",
        "        return\n",
        "\n",
        "    model = train_and_evaluate(X, y)\n",
        "    X_patient = get_patient_input(pd.DataFrame(X, columns=X.columns))\n",
        "    X_patient_scaled = preprocess(X_patient)\n",
        "    prediction = model.predict(X_patient_scaled)\n",
        "    print(f\"Prediction: {'Positive' if prediction==1 else 'Negative'}\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()\n"
      ]
    }
  ]
}