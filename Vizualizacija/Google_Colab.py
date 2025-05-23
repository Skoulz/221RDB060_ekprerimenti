import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Modelis': [
        'Svērtais – pilna', 'Svērtais – mod.',
        'Pārslēgš. – pilna', 'Pārslēgš. – mod.',
        'Iezīmes – pilna', 'Iezīmes – mod.'
    ],
    'VAK': [0.74134, 0.7877, 0.7852, 0.7903, 0.7990, 0.8164],
    'VKK': [0.9376, 0.9559, 0.9772, 1.02898, 0.9353, 1.03292],
    'Precizitāte': [0.4331, 0.3328, 0.3649, 0.2101, 0.3396, 0.20148],
    'Pārklājums': [0.80000, 0.31224, 0.2516, 0.1525, 0.2364, 0.14762]
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hibrīdo modeļu salīdzinošā analīze', fontsize=16)

# VAK
axs[0, 0].bar(df['Modelis'], df['VAK'], color='skyblue')
axs[0, 0].set_title('Vidējā absolūtā kļūda (VAK)')
axs[0, 0].tick_params(axis='x', rotation=45)

# VKK
axs[0, 1].bar(df['Modelis'], df['VKK'], color='orange')
axs[0, 1].set_title('Vidējā kvadrātiskā kļūda (VKK)')
axs[0, 1].tick_params(axis='x', rotation=45)

# Precizitāte
axs[1, 0].bar(df['Modelis'], df['Precizitāte'], color='green')
axs[1, 0].set_title('Precizitāte pret trūkstošiem datiem')
axs[1, 0].tick_params(axis='x', rotation=45)

# Pārklājums
axs[1, 1].bar(df['Modelis'], df['Pārklājums'], color='red')
axs[1, 1].set_title('Pārklājums')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()