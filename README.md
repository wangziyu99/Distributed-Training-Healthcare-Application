
# Distributed-Training-Healthcare-Application

This repository showcases a distributed framework for training machine learning models in healthcare applications, leveraging **AWS EC2**, **Hadoop**, and **PySpark** for efficient computation.

## Overview

The **Distributed Training Healthcare Application** addresses the growing demand for computational resources in healthcare. It enables scalable model training for predictive healthcare applications, such as diabetes readmission risk analysis. By utilizing distributed frameworks, the system overcomes resource limitations often faced by local healthcare systems.

### Key Features
- **Scalable Architecture**: Distributed computation across AWS EC2 instances using Spark for parallel processing and Hadoop HDFS for data management.
- **Healthcare-Specific Use Case**: Predictive analytics for hyperglycemia management and patient readmission risk.
- **User-Friendly Deployment**: Integrates a web application for personalized risk prediction.

---

## Features

### Distributed Training Framework
- **Parallel Processing**: Distributes model training tasks across multiple nodes to improve efficiency.
- **Data Management**: Reliable storage using Hadoop HDFS for large datasets.
- **Model Aggregation**: Aggregates trained models from worker nodes for centralized deployment.

### Healthcare Application
- **Predictive Analytics**: Predicts patient readmission risks based on medical history and health metrics.
- **User-Centric Design**: Provides a simple web interface for users to input health data and receive predictions.

---

## System Architecture

The system consists of two primary clusters:

1. **Distributed Model Training Cluster**:
   - Runs Spark to train machine learning models across multiple nodes.
   - Executes training tasks in parallel to optimize resource usage.
   
2. **Centralized Data Management Cluster**:
   - Uses Hadoop HDFS for distributed data storage.
   - Facilitates secure and reliable data handling during training.

### Workflow
1. Data is uploaded to the Hadoop cluster.
2. Spark retrieves the data and distributes training tasks across worker nodes.
3. Trained models are aggregated and stored in the Hadoop cluster.
4. Predictions are served via the Flask-based web application.

Refer to the architecture diagrams in the project report for detailed visualization.

---

## Technologies Used

- **AWS EC2**: For deploying virtual machines in a distributed cluster setup.
- **Hadoop**: Reliable distributed file storage via HDFS.
- **Apache Spark**: High-speed distributed computation for model training.
- **Flask**: Web application framework for user interaction.
- **Python**: Core programming language for implementation.

---

## Dataset

The system uses the **130-US Hospital Dataset** by Strack et al. (2014). Key attributes include:
- **Demographics**: Age, gender, and race.
- **Health Metrics**: Lab tests, medications, and diagnoses.
- **Outcomes**: Readmission within 30 days, readmission after 30 days, or no readmission.

The dataset's diverse attributes make it ideal for predictive modeling in healthcare.

---

## Getting Started

### Prerequisites
- AWS account with EC2 instances set up.
- Hadoop and Spark installed on the instances.
- Python environment with Flask and relevant libraries.

### Steps to Run

1. **Set Up Hadoop Cluster**:
   - Format the namenode:
     ```bash
     hadoop namenode -format
     ```
   - Start the cluster:
     ```bash
     start-dfs.sh
     hadoop start-all.sh
     ```

2. **Prepare Dataset**:
   - Upload the dataset to HDFS:
     ```bash
     hdfs dfs -mkdir /data
     hdfs dfs -put <path_to_dataset> /data/
     ```

3. **Run Spark Training**:
   - Execute training:
     ```bash
     spark-submit --total-executor-cores <num_cores> --master spark://<master_ip>:<port> model.py
     ```

4. **Deploy Web Application**:
   - Start Flask server for predictions.

---

## Results

- **Model Performance**:
  - Achieved an AUC score of 0.71 on the full feature set.
  - Achieved an AUC score of 0.60 on a reduced feature set.
- **Efficiency**: Distributed training reduced computational load, enabling scalability for larger datasets.
- **Web Application**: Provides real-time, personalized predictions through an intuitive user interface.

---

## Future Work

- **Privacy Enhancements**: Explore federated learning and differential privacy to secure patient data.
- **Optimization**: Improve scalability for larger healthcare datasets.
- **Expanded Use Cases**: Adapt the framework for chronic disease management and other applications.

---

## Citation

If you use this work, please cite:

```text
@article{your_reference_here,
  title={Distributed Training Healthcare Application},
  author={Your Name},
  journal={Project Repository},
  year={2025}
}
```

---

## License

This project is released under the MIT License:

```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions or collaboration inquiries, please contact:

**Your Name**  
**Email**: ziyuloveu@gmail.com
