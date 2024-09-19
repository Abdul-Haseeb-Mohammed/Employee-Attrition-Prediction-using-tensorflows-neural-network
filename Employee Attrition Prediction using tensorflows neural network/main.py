import pickle
import os
from src.data.make_dataset import load_data
from src.features.build_features import standardize, train_test_splitter
from src.models.train_model import set_random_seed, declare_model, compile_model, train_model, evaluate_model, create_lr_scheduler
from src.visualization.visualize import plot_training_curves, plot_lr_vs_loss, metrics_score
from src.models.predict_model import make_predictions

if __name__ == "__main__":
    # Define the base path for saving models
    base_path = "Employee Attrition Prediction using tensorflows neural network/models"
    visual_base_path = "Employee Attrition Prediction using tensorflows neural network/reports/figures"
    # Load the dataset
    data_path = "Employee Attrition Prediction using tensorflows neural network/data/raw/employee_attrition.csv"
    df = load_data(data_path)

    x_scaled = standardize(df.drop(columns=['Attrition']).copy())
    X_train, X_test, y_train, y_test = train_test_splitter(x_scaled, df['Attrition'])

    set_random_seed(42)

    # Define and train Model 1
    layers_config_1 = [{'neurons': 1, 'activation': 'linear'}]
    model_1 = declare_model(layers_config_1)
    model_1 = compile_model(model_1, loss='binary_crossentropy', optimizer='SGD')
    model_1, history_1 = train_model(model_1, X_train, y_train, epochs=5)
    loss_1, accuracy_1 = evaluate_model(model_1, X_train, y_train)
    print(f'Model 1 - Loss: {loss_1}, Accuracy: {accuracy_1}')
    plot_training_curves(history_1, "Model 1",os.path.join(visual_base_path, 'model_1_training_curves.png'))
    #plot_lr_vs_loss(history_1, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model1.png'))
    with open(os.path.join(base_path, 'model_1.pkl'), 'wb') as file:
        pickle.dump(model_1, file)

    # Define and train Model 2
    layers_config_2 = [{'neurons': 1, 'activation': 'linear'}]
    model_2 = declare_model(layers_config_2)
    model_2 = compile_model(model_2, loss='binary_crossentropy', optimizer='SGD')
    model_2, history_2 = train_model(model_2, X_train, y_train, epochs=100)
    loss_2, accuracy_2 = evaluate_model(model_2, X_train, y_train)
    print(f'Model 2 - Loss: {loss_2}, Accuracy: {accuracy_2}')
    plot_training_curves(history_2, "Model 2",os.path.join(visual_base_path, 'model_2_training_curves.png'))
    #plot_lr_vs_loss(history_2, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model2.png'))
    with open(os.path.join(base_path, 'model_2.pkl'), 'wb') as file:
        pickle.dump(model_2, file)

    # Define and train Model 3
    layers_config_3 = [
        {'neurons': 1, 'activation': 'linear'},
        {'neurons': 1, 'activation': 'linear'},
        {'neurons': 1, 'activation': 'linear'}
    ]
    model_3 = declare_model(layers_config_3)
    model_3 = compile_model(model_3, loss='binary_crossentropy', optimizer='SGD')
    model_3, history_3 = train_model(model_3, X_train, y_train, epochs=50)
    loss_3, accuracy_3 = evaluate_model(model_3, X_train, y_train)
    print(f'Model 3 - Loss: {loss_3}, Accuracy: {accuracy_3}')
    plot_training_curves(history_3, "Model 3",os.path.join(visual_base_path, 'model_3_training_curves.png'))
    #plot_lr_vs_loss(history_3, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model3.png'))
    with open(os.path.join(base_path, 'model_3.pkl'), 'wb') as file:
        pickle.dump(model_3, file)

    # Define and train Model 4
    layers_config_4 = [
        {'neurons': 2, 'activation': 'linear'},  # Hidden layer with more neurons
        {'neurons': 1, 'activation': 'linear'}   # Output layer
    ]
    model_4 = declare_model(layers_config_4)
    model_4 = compile_model(model_4, loss='binary_crossentropy', optimizer='SGD')
    model_4, history_4 = train_model(model_4, X_train, y_train, epochs=50)
    loss_4, accuracy_4 = evaluate_model(model_4, X_train, y_train)
    print(f'Model 4 - Loss: {loss_4}, Accuracy: {accuracy_4}')
    plot_training_curves(history_4, "Model 4",os.path.join(visual_base_path, 'model_4_training_curves.png'))
    #plot_lr_vs_loss(history_4, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model4.png'))
    with open(os.path.join(base_path, 'model_4.pkl'), 'wb') as file:
        pickle.dump(model_4, file)

    # Define and train Model 5
    layers_config_5 = [
        {'neurons': 1, 'activation': 'linear'},  # Hidden layer
        {'neurons': 1, 'activation': 'linear'},  # Another hidden layer
        {'neurons': 1, 'activation': 'linear'}   # Output layer
    ]
    model_5 = declare_model(layers_config_5)
    model_5 = compile_model(model_5, loss='binary_crossentropy', optimizer='SGD')
    model_5, history_5 = train_model(model_5, X_train, y_train, epochs=50)
    loss_5, accuracy_5 = evaluate_model(model_5, X_train, y_train)
    print(f'Model 5 - Loss: {loss_5}, Accuracy: {accuracy_5}')
    plot_training_curves(history_5, "Model 5",os.path.join(visual_base_path, 'model_5_training_curves.png'))
    #plot_lr_vs_loss(history_5, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model5.png'))
    with open(os.path.join(base_path, 'model_5.pkl'), 'wb') as file:
        pickle.dump(model_5, file)
    
    # Define and train Model 6
    layers_config_6 = [{'neurons': 1, 'activation': 'linear'},
                        {'neurons': 1, 'activation': 'linear'}]
    model_6 = declare_model(layers_config_6)
    model_6 = compile_model(model_6, loss='binary_crossentropy', optimizer='SGD', learning_rate=0.0009)
    model_6, history_6 = train_model(model_6, X_train, y_train, epochs=50)
    loss_6, accuracy_6 = evaluate_model(model_6, X_train, y_train)
    print(f'Model 6 - Loss: {loss_6}, Accuracy: {accuracy_6}')
    plot_training_curves(history_6, "Model 6",os.path.join(visual_base_path, 'model_6_training_curves.png'))
    #plot_lr_vs_loss(history_6, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model6.png'))
    with open(os.path.join(base_path, 'model_6.pkl'), 'wb') as file:
        pickle.dump(model_6, file)

    # Define and train Model 7
    layers_config_7 = [{'neurons': 1, 'activation': 'linear'},
                        {'neurons': 1, 'activation': 'linear'}]
    model_7 = declare_model(layers_config_7)
    model_7 = compile_model(model_7, loss='binary_crossentropy', optimizer='SGD')
    lr_scheduler = create_lr_scheduler()
    model_7, history_7 = train_model(model_7, X_train, y_train, epochs=100, callbacks=[lr_scheduler])
    loss_7, accuracy_7 = evaluate_model(model_7, X_train, y_train)
    print(f'Model 7 - Loss: {loss_7}, Accuracy: {accuracy_7}')
    plot_training_curves(history_7, "Model 7",os.path.join(visual_base_path, 'model_7_training_curves.png'))
    #plot_lr_vs_loss(history_7, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model7.png'))
    with open(os.path.join(base_path, 'model_7.pkl'), 'wb') as file:
        pickle.dump(model_7, file)

    # Define and train Model 8
    layers_config_8 = [{'neurons': 1, 'activation': 'linear'},
                        {'neurons': 1, 'activation': 'sigmoid'}]
    model_8 = declare_model(layers_config_8)
    model_8 = compile_model(model_8, loss='binary_crossentropy', optimizer='SGD', learning_rate=0.0009)
    model_8, history_8 = train_model(model_8, X_train, y_train, epochs=50)
    loss_8, accuracy_8 = evaluate_model(model_8, X_train, y_train)
    print(f'Model 8 - Loss: {loss_8}, Accuracy: {accuracy_8}')
    plot_training_curves(history_8, "Model 8",os.path.join(visual_base_path, 'model_8_training_curves.png'))
    #plot_lr_vs_loss(history_8, save_path=os.path.join(visual_base_path, 'lr_vs_loss_model8.png'))
    with open(os.path.join(base_path, 'model_8.pkl'), 'wb') as file:
        pickle.dump(model_8, file)
    