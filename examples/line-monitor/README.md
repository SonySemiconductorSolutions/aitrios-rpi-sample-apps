# Line-Monitor: Advanced Object Classification and Production Line-Monitoring

## Installation and start

### Training a custom model using Neurala Brain Builder
This example has been using the Neurala Brain Builder to train a custom classification model.
The data set used is provided in the [dataset_solderpoints.zip](./assets/dataset_solderpoints.zip).

Once the model has been trained, and exported as "Static Classifier" it has to be converted and packaged using the provided tools for the AI Camera.

Both the ```labels.txt``` and the ```network.rpk``` has to be specified in the automatically generated ```settings.json```.
For example like this:

```
...
    "AI": {
        "Model": "networks/imx500_network_solderpoints.rpk",
        "Labels": "networks/labels.txt"
    }
...
```

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ pip install -e [PATH_TO_THIS_REPO]
```

Start app:
```
$ line-monitor
```
:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

### How-to

For a quick how to guide refere to the [HOWTO](./HOWTO.md).

**Application Overview:**

The Line-Monitor is an innovative application designed to enhance efficiency and quality control in production environments. Leveraging a robust classification AI model, Line-Monitor analyzes streams of metadata to accurately identify objects passing through a production line. This intelligent system operates seamlessly with a state machine, ensuring precise and real-time analysis of each item.

**Key Features:**

1. **Object Identification and Classification:**
   - **Real-Time Analysis:** The AI model continuously monitors the production line, identifying objects with high accuracy.
   - **Classification:** Each identified object is categorized into predefined classes, ensuring streamlined sorting and processing.

2. **Quality Control:**
   - **Good/Bad Classification:** Objects are inspected and classified as good or bad based on quality criteria.
   - **Fault Detection:** For objects identified as bad, the system further subclassifies the specific type of fault, enabling targeted interventions and quality improvements.

3. **Performance Monitoring:**
   - **Period Measurement:** The application measures the time intervals for objects passing through the line, providing insights into processing speed and efficiency.
   - **Line Availability:** Line-Monitor monitors the production line status, detecting conditions such as blockage or starvation.
   - **Alerts and Notifications:** The system promptly alerts operators about any anomalies, ensuring swift resolution and minimal downtime.

**Benefits:**

- **Enhanced Efficiency:** By automating object identification and fault detection, Line-Monitor reduces the need for manual inspections, speeding up the production process.
- **Improved Quality Control:** Detailed fault classification helps in pinpointing specific issues, allowing for more precise quality management and defect reduction.
- **Real-Time Monitoring:** Continuous monitoring of line availability and performance ensures that production runs smoothly, with quick identification and resolution of any disruptions.

**Use Cases:**

- **Manufacturing:** Ideal for assembly lines, ensuring each product meets quality standards before proceeding to the next stage.
- **Packaging:** Verifies packaging integrity and correctness, classifying defective packages and identifying common issues.
- **Food and Beverage:** Ensures that products are correctly processed and packaged, identifying and categorizing any defects.

Line-Monitor transforms production line monitoring and object classification, providing a smarter, more efficient way to ensure quality and operational excellence. With its advanced AI-driven analysis, this application is a crucial tool for modern production facilities aiming for high performance and superior quality control.
