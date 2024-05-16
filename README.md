# My-Portfolio
ML

Brunch:

1) Regression_predict
Formulation of the problem:
Predict electricity consumption based on the given data:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/ba85dfbc-66bb-4a29-9b1f-e54ecf30b0b8)

Solution 1:
Add columns: Heating season (0, 1), season (-1, 1, 1.5), working days (1, -1), sine of the week, cosine of the week
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/9e1388ca-2889-42a9-aea5-426e77c4705c)

Model: LSTM, multi step multidimensional

Result:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/4bb681c8-0bbf-417f-8991-e8a90f610530)
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/05a86cc6-ef08-4762-b1a1-7e8ea3dbf4bf)

Solution 2:
Add columns: Heating season (0, 1), season (-1, 1, 1.5), working days (1, -1), sine of the week, cosine of the week, accumulative amount by day
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/6f6bf049-2dc1-4932-a52d-2193c7ede279)

Model: xgboost

Result: 
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/ce646885-3c35-4b85-a91e-c5673b295325)
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/e3e57cdf-a87f-44e3-9ea2-de47d8aa001c)
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/b497c092-66fe-4ada-9827-6110131316a4)


2) Clustering

Formulation of the problem:
It is necessary to cluster the machinery into 4 clusters (working condition, minor breakdown, major breakdown, unusable condition) based on the given data:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/f3afa8dc-02ca-47b4-bf68-e464fa755093)

Models: Kohonen neural networks, Adaptive Resonance Neural Networks, Extreme Learning Machine

Result:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/8bdf7e28-5b98-4c77-95c3-de7f03c20f9e)

Kohonen neural networks:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/2f3d403a-ccca-47b6-8dbf-914d978c18f1)
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/c72a45e7-a711-43a6-905a-de794561d55b)


3) Text classification

Formulation of the problem:
Classify reviews of smartphones and laptops, as well as positive and negative reviews.
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/699ac44e-055a-46a1-8712-d668a59bb627)

Model: CNN 1D + Bidirectional GRU

Based on sources:
Alexandre Xavier. An introduction to ConvLSTM: https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
Beakcheol Jang. Bi-LSTM Model to Increase Accuracy in Text Classification: Combining Word2vec CNN and Attention Mechanism/ Beakcheol Jang, Myeonghwi Kim, Gaspard Harerimana, Sang-ug Kang , Jong Wook Kim// Applied Sciences. 2020. : https://www.mdpi.com/journal/applsci 
CNNs for Text Classification : https://cezannec.github.io/CNN_Text_Classification/

Result:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/e071b6f6-bdb8-4f58-81ab-0cc55cdffcc6)
Final test:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/517039aa-18fb-4a8b-a612-c553298f17af)


4) Haar signs

Haar features are features of a digital image that are used in image (face) recognition.

Formulation of the problem:
Find a shape in the image:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/2f90b0a3-9182-4102-8b31-c3ede15c1b72)

Result:
![image](https://github.com/evgeniya1908/My-Portfolio/assets/49306119/b116f4b6-5ea3-4cdd-80c7-57157b30266e)
