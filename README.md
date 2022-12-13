# Intelligent Insurance Agent

~~Proof of Work project built on dummy data based on what I learned during my internship at Dawn Digitech~~

## Motive
To build an intelligent software that can act as an assistant for both:

- **Customers looking to buy insurance policies:** For customers, the assistant will predict the top 3 policies that are best fit for their needs.
- **Insurance agents selling policies:** For agents, the assistant will predict the insurance policies a potential lead is most likely to purchase along with the probability of the lead turning into an actual customer and buying the predicted policies.

## How will this help
This software will customers find the right policies from a large pool of options aggregated from the providers in the market. It will also help in reducing customer acquisition costs for insurance companies by helping the insurance agents to rank leads based on their probability of turning out be a customer and pitch them the right products.

## Technical Approach for Prototype
Based on some initial data, a Machine Learning model will be trained to predict the probability of a potential lead based on certain factors like its age, gender, income, employment type, dependents, health variables etc. and the most suitable policy for them.

This model will be built using Python and its associated libraries.

Different approaches will need to be examined to determine which one gives the highest accuracy.

The concept of Online Machine Learning will be used to ensure that the model can be continuously trained on new data points while in deployment with minimal costs. “River”, a Python library, will be used to accomplish the same.

The application thus built can be deployed as a REST API so that it has a common core that can be used on multiple platforms.

For the prototype, we will connect the API to a web frontend where the insurance agent can interact with the application.

There will be two parts to the web app:

1.	One where the agent can enter the information about the lead and get the corresponding predictions for it
2.	Second where the agent can add new data points for the Machine Learning model to train itself on
HTML and Tailwind (CSS library) will be used to build the frontend and Flask will be used for the backend.
