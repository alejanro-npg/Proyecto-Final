# %%
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from prefect import flow, task
from functools import wraps
from sqlalchemy import create_engine

# ---------------- CONFIG ----------------
DATA_DIR = "./data"
POSTGRES_CONN = "postgresql://psqluser:psqlpass@localhost:5433/bigdatatools1"
YEARS = ["2020", "2021", "2022", "2023", "2024"]

# ---------------- HELPERS ----------------
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            print(f"⏱️ Task {func.__name__} took {end - start:.2f} seconds")
    return wrapper


def clean_data(df, variable_type="GNI", method="interpolate"):
    """
    Limpieza avanzada con reglas para G20+Colombia.
    """
    g20_countries = {
        "Argentina","Australia","Brazil","Canada","China","France","Germany","India","Indonesia",
        "Italy","Japan","Mexico","Russia","Saudi Arabia","South Africa","South Korea","Turkey",
        "United Kingdom","United States","European Union","Colombia"
    }

    year_cols = [col for col in df.columns if any(str(y) in col for y in range(2020, 2026)) and variable_type in col]
    df_clean = df.copy()

    # Drop con excepción para países relevantes
    mask_relevant = df_clean["Country Name"].isin(g20_countries)
    mask_drop = (df_clean[year_cols].isna().sum(axis=1) <= 3) | mask_relevant
    df_clean = df_clean[mask_drop]

    # Métodos de relleno
    if method == "interpolate":
        df_clean[year_cols] = df_clean[year_cols].interpolate(axis=1, method="linear", limit_direction="both")
    elif method == "linear":
        df_clean[year_cols] = df_clean[year_cols].interpolate(axis=1, method="linear", limit_direction="both")
    elif method == "mean":
        row_means = df_clean[year_cols].mean(axis=1)
        df_clean[year_cols] = df_clean[year_cols].T.apply(lambda x: x.fillna(row_means)).T
    else:
        raise ValueError("Método no reconocido. Usa: interpolate, ffill, bfill o mean.")

    return df_clean.round(2)


# ---------------- TASKS -----------------
@task(name="Extract Data")
@timing_decorator
def extract_data():
    gni = pd.read_csv(os.path.join(DATA_DIR, "GNI.csv"), skiprows=3)
    gni_growth = pd.read_csv(os.path.join(DATA_DIR, "GNI_Growth.csv"), skiprows=3)
    gni_per_capita = pd.read_csv(os.path.join(DATA_DIR, "GNI_Per_Capita.csv"), skiprows=3)
    internet_pct = pd.read_csv(os.path.join(DATA_DIR, "country_internet.csv"), skiprows=3)
    internet_secure = pd.read_csv(os.path.join(DATA_DIR, "internet_secure.csv"), skiprows=3)
    broadband = pd.read_csv(os.path.join(DATA_DIR, "broadband.csv"), skiprows=3)

    print("✅ Archivos cargados correctamente")
    return gni, gni_growth, gni_per_capita, internet_pct, internet_secure, broadband


@task(name="Transform Data")
@timing_decorator
def transform_data(gni, gni_growth, gni_per_capita, internet_pct, internet_secure, broadband):
    # --- GNI ---
    df_gni = gni[["Country Name", "Country Code"] + YEARS]
    df_gni_growth = gni_growth[["Country Name", "Country Code"] + YEARS]
    df_gni_pc = gni_per_capita[["Country Name", "Country Code"] + YEARS]

    df_gni = df_gni.rename(columns={y: f"{y}_GNI" for y in YEARS})
    df_gni_growth = df_gni_growth.rename(columns={y: f"{y}_GNI_Growth" for y in YEARS})
    df_gni_pc = df_gni_pc.rename(columns={y: f"{y}_GNI_pc" for y in YEARS})

    gni_merged = df_gni.merge(df_gni_growth, on=["Country Name", "Country Code"], how="inner")
    gni_merged = gni_merged.merge(df_gni_pc, on=["Country Name", "Country Code"], how="inner")

    # --- Internet ---
    internet_pct = internet_pct[["Country Name", "Country Code"] + YEARS]
    internet_secure = internet_secure[["Country Name", "Country Code"] + YEARS]
    broadband = broadband[["Country Name", "Country Code"] + YEARS]

    internet_pct = internet_pct.rename(columns={y: f"{y}_InternetPct" for y in YEARS})
    internet_secure = internet_secure.rename(columns={y: f"{y}_SecureServers" for y in YEARS})
    broadband = broadband.rename(columns={y: f"{y}_Broadband" for y in YEARS})

    internet_merged = internet_pct.merge(internet_secure, on=["Country Name", "Country Code"], how="outer")
    internet_merged = internet_merged.merge(broadband, on=["Country Name", "Country Code"], how="outer")

    # --- Limpieza avanzada ---
    gni_final = clean_data(gni_merged, variable_type="GNI", method="interpolate")
    internet_final = clean_data(internet_merged, variable_type="_InternetPct", method="interpolate")
    internet_final = clean_data(internet_final, variable_type="_Broadband", method="linear")
    internet_final = clean_data(internet_final, variable_type="_SecureServers", method="interpolate")

    print("✅ Transformación + Limpieza completa")
    return gni_final, internet_final


@task(name="Analyze Data")
@timing_decorator
def analyze_data(gni_df, internet_df):
    # --- Missing values ---
    plt.figure(figsize=(12, 6))
    gni_df.isna().sum().plot(kind="bar", color="red", alpha=0.7)
    plt.title("Valores faltantes en GNI")
    plt.tight_layout()
    plt.savefig("missing_gni.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    internet_df.isna().sum().plot(kind="bar", color="blue", alpha=0.7)
    plt.title("Valores faltantes en Internet")
    plt.tight_layout()
    plt.savefig("missing_internet.png")
    plt.close()

    # --- Correlación ---
    merged = gni_df.merge(internet_df, on=["Country Name", "Country Code"], how="inner")
    corr = merged.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matriz de correlación GNI + Internet")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()
    YEARS = ["2020", "2021", "2022", "2023", "2024"]
    # =====================================================
    # 📊 NUEVAS GRÁFICAS DE ANÁLISIS EXPLORATORIO
    # =====================================================
    years = [int(y) for y in YEARS]

    # --- 1. Evolución temporal: Internet vs GNI per capita ---
 # Definimos los grupos
    secciones = {
        "Desarrollados": ["United States", "Germany", "Japan"],
        "EnDesarrollo": ["India", "Brazil", "Colombia"],
        "Subdesarrollados": ["Ethiopia", "Uganda", "Mozambique"]
    }
    
    colores = ["orange", "blue", "green"]

    for nombre_seccion, lista_paises in secciones.items():
        fig, ax1 = plt.subplots(figsize=(10, 6))

        for i,country in enumerate(lista_paises):
            if country in merged["Country Name"].values:
                df_country = merged[merged["Country Name"] == country]

                # GNI per capita (naranja, línea sólida)
                ax1.plot(
                    years,
                    df_country[[f"{y}_GNI_pc" for y in YEARS]].values.flatten(),
                    marker="o",
                     color=colores[i],
                    linestyle="-",
                    label=f"{country} - GNIpc"
                )

        ax1.set_xlabel("Año")
        ax1.set_ylabel("GNI per capita (USD)", color="orange")
        ax1.tick_params(axis="y", labelcolor="orange")

        # Segundo eje para Internet % (azul, punteado)
        ax2 = ax1.twinx()
        for i,country in enumerate(lista_paises):
            if country in merged["Country Name"].values:
                df_country = merged[merged["Country Name"] == country]

                ax2.plot(
                    years,
                    df_country[[f"{y}_InternetPct" for y in YEARS]].values.flatten(),
                    marker="s",
                     color=colores[i],
                    linestyle="--",
                    
                )

        ax2.set_ylabel("% Internet Users", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.title(f"Evolución GNIpc vs Internet en {nombre_seccion}")
        fig.tight_layout()
        plt.savefig(f"evolucion_gni_internet_{nombre_seccion}.png")
        plt.close()

    # --- 2. Barras por país: Broadband vs Secure Internet ---
    ultimo_anio = YEARS[-1]
  
    # Construcción del dataframe con país y sección
    registros = []
    for nombre_seccion, lista_paises in secciones.items():
        for pais in lista_paises:
            fila = merged[merged["Country Name"] == pais]
            if not fila.empty:
                broadband = fila[f"{ultimo_anio}_Broadband"].values[0]
                secure = fila[f"{ultimo_anio}_SecureServers"].values[0]
                registros.append({
                    "País": pais,
                    "Sección": nombre_seccion,
                    "Broadband": broadband,
                    "Secure Internet": secure
                })

    df_bar = pd.DataFrame(registros)

    # Ordenar países por sección
    df_bar["Orden"] = df_bar["Sección"].map({"Desarrollados": 0, "En Desarrollo": 1, "Subdesarrollados": 2})
    df_bar = df_bar.sort_values(["Orden", "País"])

    # Crear gráfico de doble eje
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eje 1: Broadband
    ancho = 0.4
    posiciones = range(len(df_bar))
    ax1.bar([p - ancho/2 for p in posiciones], df_bar["Broadband"], 
            width=ancho, label="Broadband", color="tab:orange")
    ax1.set_ylabel("Broadband per 100 people")
    ax1.set_ylim(0, 50)

    # Eje 2: Secure Internet
    ax2 = ax1.twinx()
    ax2.bar([p + ancho/2 for p in posiciones], df_bar["Secure Internet"], 
            width=ancho, label="Secure Internet", color="tab:blue")
    ax2.set_ylabel("Secure Internet per million people")

    # Configuración del eje X
    ax1.set_xticks(posiciones)
    ax1.set_xticklabels(df_bar["País"], rotation=45, ha="right")

    # Título y leyenda combinada
    fig.suptitle(f"Broadband vs Secure Internet por País ({ultimo_anio})", fontsize=14)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left")

    plt.tight_layout()
    plt.savefig("barras_broadband_secure_doble.png")
    plt.close()


    # --- 3. Scatter Internet vs GNI per capita ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=merged,
        x="2024_InternetPct", y="2024_GNI_pc",
        hue="Country Name", alpha=0.7, legend=False
    )
    sns.regplot(
        data=merged, x="2024_InternetPct", y="2024_GNI_pc",
        scatter=False, color="red"
    )
    plt.title("Relación entre % Usuarios de Internet y GNI per capita (2024)")
    plt.xlabel("% Usuarios de Internet")
    plt.ylabel("GNI per capita")
    plt.tight_layout()
    plt.savefig("scatter_internet_vs_gnipc_2024.png")
    plt.close()

    # --- 4. Tasas de crecimiento comparadas ---
    growth_df = merged.copy()
    for var in ["GNI_pc", "InternetPct"]:
        cols = [f"{y}_{var}" for y in YEARS]
        growth_df[f"{var}_growth"] = growth_df[cols].pct_change(axis=1).mean(axis=1) * 100

    plt.figure(figsize=(10, 6))
    sns.histplot(growth_df["GNI_pc_growth"], kde=True, color="orange", label="GNIpc Growth")
    sns.histplot(growth_df["InternetPct_growth"], kde=True, color="blue", label="Internet Growth")
    plt.legend()
    plt.title("Distribución de crecimiento medio (%) - GNIpc vs Internet")
    plt.tight_layout()
    plt.savefig("growth_distribution_gnipc_internet.png")
    plt.close()

    # =====================================================
    # --- Clustering (ejemplo: GNI per capita 2024 vs InternetPct 2024) ---
    cluster_df = merged[["Country Name", "2024_GNI_pc", "2024_InternetPct"]].dropna()
    X = cluster_df[["2024_GNI_pc", "2024_InternetPct"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 8), inertias, marker="o")
    plt.title("Método del Codo - GNIpc vs InternetPct")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.savefig("elbow_clusters.png")
    plt.close()

    # KMeans final con k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["Cluster"] = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # Silhouette
    score = silhouette_score(X_scaled, cluster_df["Cluster"])
    print(f"📊 Silhouette Score (k=3): {score:.4f}")

    # Scatter con centroides
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(cluster_df["2024_GNI_pc"], cluster_df["2024_InternetPct"],
                          c=cluster_df["Cluster"], cmap="viridis", alpha=0.7)
    plt.scatter(scaler.inverse_transform(centroids)[:, 0],
                scaler.inverse_transform(centroids)[:, 1],
                c="red", marker="X", s=200, label="Centroids")
    plt.xlabel("GNI per capita (2024)")
    plt.ylabel("% Internet Users (2024)")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.legend()
    plt.title("Clusters: GNIpc vs InternetPct (2024)")
    plt.savefig("clusters_scatter.png")
    plt.close()

    return cluster_df

@task(name="Load Data")
@timing_decorator
def load_data(gni_df, internet_df, cluster_df):
    engine = None
    try:
        engine = create_engine(POSTGRES_CONN)
        gni_df.to_sql("gni_data", engine, if_exists="replace", index=False)
        internet_df.to_sql("internet_data", engine, if_exists="replace", index=False)
        cluster_df.to_sql("gni_internet_clusters", engine, if_exists="replace", index=False)
        print("✅ Datos cargados exitosamente en PostgreSQL")
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando datos: {e}")
    finally:
        if engine:
            engine.dispose()


# ---------------- FLOW -----------------
@flow(name="ETL GNI + Internet + Clusters")
def etl_gni_internet():
    gni, gni_growth, gni_pc, internet_pct, internet_secure, broadband = extract_data()
    gni_df, internet_df = transform_data(gni, gni_growth, gni_pc, internet_pct, internet_secure, broadband)
    cluster_df = analyze_data(gni_df, internet_df)
    load_data(gni_df, internet_df, cluster_df)


if __name__ == "__main__":
    etl_gni_internet()


# %% [markdown]
# # Conclusiones del análisis GNIpc vs Internet
# 
# ## 1. Relación entre Internet y GNIpc
# - Existe una **correlación positiva y significativa**: a mayor porcentaje de usuarios de Internet, mayor es el GNI per cápita.
# - Aunque no es una relación perfectamente lineal, la tendencia confirma que la **digitalización está asociada al desarrollo económico**.
# - Los países con bajo acceso a Internet suelen presentar también un bajo GNIpc.
# 
# ---
# 
# ## 2. Clustering (KMeans con 3 grupos)
# - Se identificaron **tres clusters principales de países**:
#   - **Cluster bajo** → GNIpc reducido y baja penetración de Internet (países en vías de desarrollo).
#   - **Cluster medio** → ingresos y acceso digital en crecimiento (economías emergentes).
#   - **Cluster alto** → GNIpc elevado y casi saturación en acceso a Internet (economías desarrolladas).
# - El método del codo validó que **k=3 es una segmentación razonable**.
# 
# ---
# 
# ## 3. Evolución temporal (2020-2024)
# - En **China, Colombia y EE.UU.** el GNIpc crece de manera sostenida.
# - El **% de usuarios de Internet crece más lentamente** en países desarrollados debido a la saturación.
# - En economías emergentes, como Colombia y China, se observa que la **expansión digital acompaña el crecimiento económico**, sugiriendo un rol potenciador de Internet.
# 
# ---
# 
# ## 4. Matriz de correlación
# - El **GNIpc y el % de usuarios de Internet tienen correlaciones altas y positivas** en todos los años analizados.
# - También existen correlaciones con variables como **banda ancha** y **servidores seguros**, lo que refuerza que la **infraestructura digital impulsa el desarrollo económico**.
# 
# ---
# 
# ## 5. Distribución de crecimiento (%)
# - El **crecimiento de Internet es más disperso y volátil**, con países que avanzan rápido y otros que se estancan.
# - El **crecimiento del GNIpc es más estable y concentrado**.
# - Esto muestra que la **brecha digital es más marcada que la económica** en el corto plazo.
# 
# ---
# 
# ## 6. Método del codo
# - El análisis mostró un quiebre claro en **k=3**, confirmando la existencia de tres niveles principales de desarrollo digital-económico.
# 
# ---
# 
# ## Conclusión general
# Los resultados confirman que existe una **relación directa entre digitalización y desarrollo económico**.  
# - Los países con **mayor penetración de Internet** tienden a tener **mayores ingresos y mayor estabilidad económica**.  
# - Los países con **bajo acceso digital** permanecen en niveles bajos de GNIpc.  
# - La **brecha digital refleja también una brecha económica**, evidenciando la importancia de la inversión en infraestructura tecnológica para impulsar el crecimiento.
# 


