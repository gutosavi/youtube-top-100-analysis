# Instalações

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from ydata_profiling import ProfileReport

sns.set(style='whitegrid', rc={'figure.dpi':100})
pd.set_option('display.max_colwidth', 200)


df = pd.read_csv(r'd:/Python/top_youtube/data/youtube_top_100_songs_2025.csv')
profile = ProfileReport(df, title="Profiling Report")

df.info()
print()
df.head()

# ---

df.columns


# Dominância no YouTube: Top Artistas/Canais

coluna_visualizacoes = 'view_count'
coluna_canal = 'channel'

print("\nTOP 5 CANAIS POR TOTAL DE VISUALIZAÇÕES")

dominancia = df.groupby(coluna_canal)[coluna_visualizacoes].sum()
top_canais = dominancia.sort_values(ascending=False).head(5)

top_canais_df = top_canais.reset_index()
top_canais_df.columns = ['Canal', 'Total de Visualizações']

print(top_canais_df.to_string(index=False))


# Top 10 músicas mais vistas

top10 = df.nlargest(10, 'view_count').sort_values('view_count')

plt.figure(figsize=(10,6))
sns.barplot(x='view_count', y='title', data=top10, orient='h')
plt.xlabel('Visualizações')
plt.ylabel('')
plt.title('Top 10 músicas por visualizações (YouTube Top 100 - 2025)')
plt.ticklabel_format(axis='x', style='plain')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x>=1e6 else f'{int(x/1e3)}K' if x>=1e3 else f'{int(x)}'))
plt.tight_layout()
plt.savefig('top10_views.png', dpi=150)
plt.show()


# Top canais por número de vídeos - ver quem aparece mais vezes

top_channels_count = df['channel'].value_counts().head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_channels_count.values, y=top_channels_count.index)
plt.xlabel('Número de vídeos no Top 100')
plt.title('Canais com mais vídeos no Top 100')
plt.tight_layout()
plt.savefig('top_channels_count.png', dpi=150)
plt.show()

top_channels_count.reset_index().rename(columns={'count':'Contagem','channel':'Artista'})


# Canais: total de views e média por vídeo comparando poder de audiência (soma vs média).

agg = df.groupby('channel').agg(
    n_videos = ('title','size'),
    total_views = ('view_count','sum'),
    mean_views = ('view_count','mean'),
    followers = ('channel_follower_count','max')
).reset_index()

top_by_total = agg.nlargest(10, 'total_views').sort_values('total_views')
top_by_total.rename(columns={'channel': 'Canal'}, inplace=True)
top_by_total.rename(columns={'total_views': 'Total de Visualizações'}, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x='Total de Visualizações', y='Canal', data=top_by_total)

plt.title('Top 10 canais por total de visualizações (soma)')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x>=1e6 else f'{int(x/1e3)}K' if x>=1e3 else f'{int(x)}'))
plt.tight_layout()
plt.savefig('top_channels_total_views.png', dpi=150)


# Views vs Inscritos (followers) — dispersão com escala log - ver se mais inscritos = mais views (regra geral) e identificar outliers.

agg_plot = agg.copy()
agg_plot = agg_plot[agg_plot['followers']>0]

plt.figure(figsize=(8,6))
plt.scatter(agg_plot['followers'], agg_plot['mean_views'], s=60, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Seguidores do canal (log scale)')
plt.ylabel('Visualizações médias por vídeo (log scale)')
plt.title('Followers x Mean Views (por canal) — escala log')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig('followers_vs_meanviews.png', dpi=150)
plt.show()

# Correlação (Pearson) nos logs
corr = np.corrcoef(np.log1p(agg_plot['followers']), np.log1p(agg_plot['mean_views']))[0,1]
print("Correlação (Pearson) entre log(followers) e log(mean_views):", round(corr,3))


# Top 5 canais pequenos canais que bombaram - views por followers

agg['views_per_follower'] = agg['total_views'] / agg['followers'].replace(0, np.nan)
top_ppf = agg.sort_values('views_per_follower', ascending=False).head(5)
top_ppf[['channel','n_videos','total_views','followers','views_per_follower']]


# Colaborações vs Solo comparando médias e testando diferença estatística.

solo = df[~df['collab']]
collab = df[df['collab']]

print("Média views - solo:", int(solo['view_count'].mean()))
print("Média views - collab:", int(collab['view_count'].mean()))
print("Mediana views - solo:", int(solo['view_count'].median()))
print("Mediana views - collab:", int(collab['view_count'].median()))


# Teste estatístico (Mann-Whitney é robusto para distribuições assimétricas)
from scipy.stats import mannwhitneyu
u_stat, p_val = mannwhitneyu(collab['view_count'], solo['view_count'], alternative='two-sided')
print("Mann-Whitney U p-value:", p_val)

print()

if p_val < 0.05:
    print("Há evidência estatística de diferença.")
else:
    print("Não há evidência estatística de diferença.")


# Duração x Views — análise por faixas, ver se existe “duração ideal”.

df['minutos'] = df['duration'] / 60

bins = [0, 2.5, 3.5, 5, 15] 
labels = ['pequeno (<=2.5m)','médio (2.5-3.5m)','longo (3.5-5m)','muito longo (>5m)']
df['duration_bin'] = pd.cut(df['minutos'], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(9,5))
sns.boxplot(x='duration_bin', y='view_count', data=df)
plt.yscale('log') 
plt.title('Duração vs Visualizações')
plt.xlabel('')
plt.tight_layout()
plt.savefig('duration_vs_views_boxplot.png', dpi=150)
plt.show()

# Médias por bin
df.groupby('duration_bin')['view_count'].agg(['count','mean','median']).reset_index()


# Resumo gerado pela bibliteca ydata-profiling - apenas para testar a ferramenta

profile.to_notebook_iframe()

