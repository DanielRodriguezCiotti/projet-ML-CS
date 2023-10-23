
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import chardet
import random
import os

#Pour tester le modèle sans les nouvelles features, mettre en commentaire les lignes 28 à 34 et supprimer 'covid','Noël','vacances' de colonnes_a_inclure ligne 19.

# Trouver les données
script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, '..', 'data', 'preprocessed_tgv_data.csv')

# Lire un morceau du fichier pour détecter l'encodage
with open(data_file_path, 'rb') as f:
    result = chardet.detect(f.read(10000))  # Lire les premiers 10 000 octets pour la détection

# Afficher l'encodage détecté
print('Encodage détecté:', result['encoding'])

# Liste des noms des colonnes à inclure
colonnes_a_inclure = ['date','service','gare_depart','gare_arrivee','nb_train_prevu','nb_annulation','nb_train_depart_retard','retard_moyen_depart','retard_moyen_tous_trains_depart','nb_train_retard_arrivee','retard_moyen_arrivee','retard_moyen_tous_trains_arrivee','nb_train_retard_sup_15','retard_moyen_trains_retard_sup15','nb_train_retard_sup_30','nb_train_retard_sup_60','prct_cause_externe','prct_cause_infra','prct_cause_gestion_trafic','prct_cause_materiel_roulant','prct_cause_gestion_gare','prct_cause_prise_en_charge_voyageurs','duree_standard','covid','Noël','vacances']
# Remplacez 'chemin_vers_votre_fichier.xlsx' par le chemin vers votre fichier Excel
df = pd.read_csv(data_file_path, engine='c', sep=';', usecols=colonnes_a_inclure, encoding=result['encoding'])
df_encoded = pd.get_dummies(df, columns=['gare_depart', 'gare_arrivee'], drop_first=True)

# Conversion de 'service'
df_encoded['service'] = df_encoded['service'].apply(lambda x: 1 if x == 'National' else 0)
 
# Conversion de 'Période Noël'
df_encoded['Noël'] = df_encoded['Noël'].apply(lambda x: 1 if x == 'VRAI' else 0)

# Conversion de 'Période COVID'
df_encoded['covid'] = df_encoded['covid'].apply(lambda x: 2 if x == 'VRAI + PASS' else (1 if x == 'VRAI' else 0))

# Conversion de 'Période Vacances'
df_encoded['vacances'] = df_encoded['vacances'].apply(lambda x: 1 if x == 'VRAI' else 0)
 
start_date = pd.to_datetime('2018-01')
end_date = pd.to_datetime('2022-12')
df_encoded['date'] = pd.to_datetime(df_encoded['date'])

df_test = df_encoded[(df_encoded['date'] >= start_date) & (df_encoded['date'] <= end_date)]

colonnes_a_predire = ['retard_moyen_arrivee','retard_moyen_tous_trains_arrivee','prct_cause_externe','prct_cause_infra','prct_cause_gestion_trafic','prct_cause_materiel_roulant','prct_cause_gestion_gare','prct_cause_prise_en_charge_voyageurs']
y = df_test[colonnes_a_predire]
colonnes_a_exclure = ['date','nb_annulation','nb_train_depart_retard','retard_moyen_depart','retard_moyen_tous_trains_depart','nb_train_retard_arrivee','retard_moyen_arrivee','retard_moyen_tous_trains_arrivee','nb_train_retard_sup_15','retard_moyen_trains_retard_sup15','nb_train_retard_sup_30','nb_train_retard_sup_60','prct_cause_externe','prct_cause_infra','prct_cause_gestion_trafic','prct_cause_materiel_roulant','prct_cause_gestion_gare','prct_cause_prise_en_charge_voyageurs']
X = df_test.drop(columns=colonnes_a_exclure)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# Calcul de l'erreur (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error globale sur les données de test :', mse)

# Calcul de l'erreur (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error globale sur les données de test :', mae)
for i in range(8):
    # Calcul de l'erreur (Mean Squared Error)
    mse_2023 = mean_squared_error(y_test.iloc[:,i], y_pred[:,i])
    print('Mean Squared Error sur les données de test (colonne' + str(i) +'):', mse_2023)

    # Calcul de l'erreur (Mean Absolute Error)
    mae_2023 = mean_absolute_error(y_test.iloc[:,i], y_pred[:,i])
    print('Mean Absolute Error sur les données de test (colonne' + str(i) +'):', mae_2023)

# Data de 2023
start_date = pd.to_datetime('2023-01')
df_2023 = df_encoded[(df_encoded['date'] >= start_date)]
X_2023 = df_2023.drop(columns=colonnes_a_exclure)

# Prédiction avec le modèle
y_pred_2023 = rf_model.predict(X_2023)
y_reel_2023 = df_2023[colonnes_a_predire]

mse_2023 = mean_squared_error(y_reel_2023, y_pred_2023)
print('Mean Squared Error globale sur les données de 2023 :', mse_2023)

# Calcul de l'erreur (Mean Absolute Error)
mae_2023 = mean_absolute_error(y_reel_2023, y_pred_2023)
print('Mean Absolute Error globale sur les données de 2023 :', mae_2023)

for i in range(8):
    # Calcul de l'erreur (Mean Squared Error)
    mse_2023 = mean_squared_error(y_reel_2023.iloc[:,i], y_pred_2023[:,i])
    print('Mean Squared Error pour 2023 (colonne' + str(i) +'):', mse_2023)

    # Calcul de l'erreur (Mean Absolute Error)
    mae_2023 = mean_absolute_error(y_reel_2023.iloc[:,i], y_pred_2023[:,i])
    print('Mean Absolute Error pour 2023 (colonne' + str(i) +'):', mae_2023)

#afficher les prédictions et les vraies valeurs pour une comparaison visuelle
import matplotlib.pyplot as plt
indices_a_plot_hasard = [random.randint(1, 100) for _ in range(20)]
indices_a_plot_hasard = sorted(indices_a_plot_hasard)
indices_a_plot = [10, 12, 38, 68, 78, 85, 97, 110, 115, 121, 131, 140, 149, 157, 174, 184, 201, 204, 212, 293]
plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,0], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,0], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('retard_moyen_arrivee')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('retard_moyen_arrivee.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,1], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,1], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('retard_moyen_tous_trains_arrivee')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('retard_moyen_tous_trains_arrivee.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,2], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,2], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_externe')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_externe.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,3], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,3], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_infra')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_infra.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,4], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,4], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_gestion_trafic')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_gestion_trafic.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,5], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,5], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_materiel_roulant')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_materiel_roulant.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,6], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,6], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_gestion_gare')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_gestion_gare.png')

plt.figure(figsize=(12, 6))
plt.plot(indices_a_plot, y_reel_2023.iloc[indices_a_plot,7], label='Vraies valeurs')
plt.plot(indices_a_plot, y_pred_2023[indices_a_plot,7], label='Prédictions')
plt.legend()
plt.xlabel('Index')
plt.ylabel('prct_cause_prise_en_charge_voyageurs')
plt.title('Comparaison des prédictions avec les vraies valeurs pour 2023')
plt.show()
#plt.savefig('prct_cause_prise_en_charge_voyageurs.png')
