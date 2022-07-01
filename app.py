import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import plotly.express as px
# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
                   layout='wide')


# ---------------------------------#
# Model building
def build_model(df):
    X = df.drop(columns=['which_kind', 'likely_to_make_another_purchase', 'likely_to_switch_to_another_brand',
                          'continue_using_after_increase_in_price_new', 'Unnamed: 24', 'willing_to_recommend_new',
                          'seen_advertisements', 'impression', 'willing_to_recommend', 'ads_description', 'factors',
                          'familiarity', 'gender', 'rate_itc_compared_to_other_brands',
                          'buy_itc_products_based_on_advertisements', 'price_rating', 'quality_rating',
                          'features_rating', 'brand_value_rating', 'availability_rating', 'value_of_money_rating',
                          'brand_promotion_rating', 'experience_with_products_rating', 'duration_of_use',
                          'continue_using_after_increase_in_price', 'likely_to_make_another_purchase',
                          'likely_to_switch_to_another_brand'])
    Y = df['willing_to_recommend_new']
    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    model_xgboost_fin = xgboost.XGBClassifier(n_estimators=parameter_n_estimators, learning_rate=parameter_learning_rate,
    colsample_bylevel =parameter_colsample_bylevel,
    max_depth=parameter_max_depth,
    min_child_weight=parameter_min_child_weight,
    gamma=parameter_gamma,
    subsample=parameter_subsample,
    colsample_bytree=parameter_colsample_bytree,
    use_label_encoder=parameter_use_label_encoder,
    verbosity = 1)
    eval_set = [(X_train, Y_train), (X_test, Y_test)]

    model_xgboost_fin.fit(X_train,
                          Y_train, early_stopping_rounds=20,
                  eval_set=eval_set,
                  verbose=True)

    def model_prediction(input_data):
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asfarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model_xgboost_fin.predict(input_data_reshaped)
        print(prediction)
        return prediction

    st.subheader('2. Graphs')
    st.markdown('**2.1. Familiarity with brand**')
    labels = df.familiarity.value_counts().index
    sizes = df.familiarity.value_counts().values
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
    st.markdown('**2.2. Effectiveness of advertisement**')
    labels = df.ads_effective.value_counts().index
    sizes = df.ads_effective.value_counts().values
    dict = {'ads_effective': labels,
            'Val_count': sizes}

    df1 = pd.DataFrame(dict)
    fig = px.bar(
        df1,
        x="ads_effective",
        y="Val_count",
        title="Effectiveness of advertisement"
    )
    st.plotly_chart(fig)
    st.markdown('**2.3. The period of use of classmate product**')
    labels = df.duration_of_use.value_counts().index
    sizes = df.duration_of_use.value_counts().values
    dict = {'duration_of_use': labels,
            'Val_count': sizes}

    df1 = pd.DataFrame(dict)
    fig = px.bar(
        df1,
        x="duration_of_use",
        y="Val_count",
        title="The period of use of classmate product"
    )
    st.plotly_chart(fig)

    st.subheader('3. Model Performance')
    st.markdown('**3.1. Test set**')

    def main():
        age = st.text_input('age')
        ads_effective = st.text_input('ads_effective')
        fam_new = st.text_input('fam_new')
        gen_new = st.text_input('gen_new')
        rate_new = st.text_input('rate_new')
        buy_new = st.text_input('buy_new')
        price_new = st.text_input('price_new')
        quality_new = st.text_input('quality_new')
        features_new = st.text_input('features_new')
        brand_value_new = st.text_input('brand_value_new')
        availability_new = st.text_input('availability_new')
        value_new = st.text_input('value_new')
        brand_promotions_new = st.text_input('brand_promotions_new')
        experience_new = st.text_input('experience_new')
        duration_new = st.text_input('duration_new')
        seen_advertisements_new = st.text_input('seen_advertisements_new')

        # code for Prediction
        diagnosis = ''

        # creating a button for Prediction

        if st.button('Test result'):
            diagnosis = model_prediction([age, ads_effective, fam_new, gen_new, rate_new, buy_new, price_new, quality_new, features_new, brand_value_new, availability_new, value_new, brand_promotions_new, experience_new, duration_new, seen_advertisements_new])


        st.success(diagnosis)

    if __name__ == '__main__':
        main()
    st.subheader('4. Model Parameters')
    st.write(model_xgboost_fin.get_params())


# ---------------------------------#
st.write("""
# The Machine Learning App

In this implementation, the *Xgboost()* function is used in this app for build a regression model using the **Xgboost** algorithm.

Try adjusting the hyperparameters!

""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](C:/Users/Abhi/Desktop/ITC/new.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 900, 100)
    parameter_min_child_weight = st.sidebar.slider('Min child weight (min_child_weight)', 1, 15, 4, 1)
    parameter_learning_rate = st.sidebar.slider(
        'Learning rate (learning_rate)', 0.0, 0.5, 0.16, 0.05)
    parameter_max_depth = st.sidebar.slider(
        'Max_depth (max_depth)', 1, 15, 1, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_gamma = st.sidebar.slider('Gamma (gamma)', 0, 10, 2, 1)
    parameter_subsample = st.sidebar.slider('Subsample (subsample)', 0.0, 1.00, 0.35, 0.05)
    parameter_colsample_bylevel = st.sidebar.slider('Colsample bylevel (colsample_bylevel)',
                                                   0.00, 1.00, 0.95, 0.05)
    parameter_colsample_bytree = st.sidebar.slider('Colsample bytree (colsample_bytree)',
                                                           0.00, 1.00, 0.80, 0.05)
    parameter_use_label_encoder = st.sidebar.select_slider('Use label encoder (n_jobs)', options=[False, True])

# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        new = pd.read_csv('C:/Users/Abhi/Desktop/ITC/new.csv')
        X = new.drop(columns=['which_kind', 'likely_to_make_another_purchase', 'likely_to_switch_to_another_brand',
                              'continue_using_after_increase_in_price_new', 'Unnamed: 24', 'willing_to_recommend_new',
                              'seen_advertisements', 'impression', 'willing_to_recommend', 'ads_description', 'factors',
                              'familiarity', 'gender', 'rate_itc_compared_to_other_brands',
                              'buy_itc_products_based_on_advertisements', 'price_rating', 'quality_rating',
                              'features_rating', 'brand_value_rating', 'availability_rating', 'value_of_money_rating',
                              'brand_promotion_rating', 'experience_with_products_rating', 'duration_of_use',
                              'continue_using_after_increase_in_price', 'likely_to_make_another_purchase',
                              'likely_to_switch_to_another_brand'])
        Y = new['willing_to_recommend_new']
        df = pd.concat([X, Y], axis=1)

        st.markdown('The ITC classmate dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)
