import os
import urllib.request
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    InceptionV3,
    EfficientNetB0,
    MobileNetV3Large
)

MODEL_URL = "https://github.com/adliaw326/Klasifikasi-daun-cabai/releases/download/v1.0.0/cnn_phenotype_model_skenario3_tuned.h5"

# ==============================================================================
# 1. PILIH SKENARIO
# ==============================================================================
#
# GANTI ANGKA INI UNTUK MENGGANTI MODEL YANG DIGUNAKAN STREAMLIT
#
# 3 = InceptionV3
# 4 = EfficientNetB0
# 5 = MobileNetV3Large
#
# ==============================================================================

SELECTED_SCENARIO = 3


# ==============================================================================
# 2. KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Clustering Fenotipe Daun Cabai",
    page_icon="🌶️",
    layout="centered"
)


# ==============================================================================
# 3. KONFIGURASI MODEL
# ==============================================================================

MODEL_CONFIG = {

    # --------------------------------------------------------------------------
    # SKENARIO 3 - INCEPTIONV3
    # --------------------------------------------------------------------------

    3: {
        "name": "InceptionV3",
        "model_path": "cnn_phenotype_model_skenario3_tuned.h5",
        "input_size": (299, 299),
        "preprocess": (
            tf.keras.applications.inception_v3.preprocess_input
        ),
        "model_type": "full_model",
        "num_classes": 4
    },


    # --------------------------------------------------------------------------
    # SKENARIO 4 - EFFICIENTNETB0
    # --------------------------------------------------------------------------

    4: {
        "name": "EfficientNetB0",
        "model_path": "cnn_phenotype_model_skenario4_tuned.h5",
        "input_size": (224, 224),
        "preprocess": (
            tf.keras.applications.efficientnet.preprocess_input
        ),
        "model_type": "weights_only",
        "num_classes": 4
    },


    # --------------------------------------------------------------------------
    # SKENARIO 5 - MOBILENETV3LARGE
    # --------------------------------------------------------------------------

    5: {
        "name": "MobileNetV3Large",
        "model_path": "cnn_phenotype_model_skenario5_tuned.h5",
        "input_size": (224, 224),
        "preprocess": (
            tf.keras.applications.mobilenet_v3.preprocess_input
        ),
        "model_type": "weights_only",
        "num_classes": 4
    }
}


# ==============================================================================
# 4. VALIDASI SKENARIO
# ==============================================================================

if SELECTED_SCENARIO not in MODEL_CONFIG:

    raise ValueError(
        "SELECTED_SCENARIO harus bernilai 3, 4, atau 5."
    )


config = MODEL_CONFIG[SELECTED_SCENARIO]


# ==============================================================================
# 5. HYPERPARAMETER ARSITEKTUR
# ==============================================================================

# ------------------------------------------------------------------------------
# SKENARIO 4
# ------------------------------------------------------------------------------

SCENARIO_4_PARAMS = {

    "dropout": 0.7,

    "dense_units": 512,

    "fine_tune_layers": 30
}


# ------------------------------------------------------------------------------
# SKENARIO 5
# ------------------------------------------------------------------------------
#
# GANTI DENGAN BEST HYPERPARAMETER SEBENARNYA DARI HASIL TUNING SKENARIO 5
# ------------------------------------------------------------------------------

SCENARIO_5_PARAMS = {

    "dropout": 0.3,

    "dense_units": 256,

    "fine_tune_layers": 20
}


# ==============================================================================
# 6. NAMA CLUSTER
# ==============================================================================

CLASS_NAMES = [

    "Cluster 0",

    "Cluster 1",

    "Cluster 2",

    "Cluster 3"
]


# ==============================================================================
# 7. DESKRIPSI CLUSTER
# ==============================================================================

# CLUSTER_DETAILS = {

#     "Cluster 0":
#         "Karakteristik: Daun berukuran sempit-memanjang "
#         "(lanceolate), tepi daun relatif rata, dengan "
#         "guratan vena yang halus.",

#     "Cluster 1":
#         "Karakteristik: Daun berukuran lebar-oval "
#         "(ovate), ujung daun meruncing, dengan tekstur "
#         "permukaan yang relatif lebih kasar.",

#     "Cluster 2":
#         "Karakteristik: Daun berukuran sedang dengan "
#         "rasio panjang dan lebar relatif seimbang, "
#         "warna hijau tua, serta helai daun cenderung melengkung.",

#     "Cluster 3":
#         "Karakteristik: Daun berukuran relatif besar "
#         "dengan geometri helai yang melebar dan "
#         "jarak antar-vena sekunder yang relatif renggang."
# }


# ==============================================================================
# 8. MEMBANGUN MODEL SKENARIO 4
# ==============================================================================

def build_scenario_4():

    params = SCENARIO_4_PARAMS

    base_model = EfficientNetB0(

        weights="imagenet",

        include_top=False,

        input_shape=(224, 224, 3)
    )


    base_model.trainable = True


    for layer in base_model.layers[
        :-params["fine_tune_layers"]
    ]:

        layer.trainable = False


    inputs = layers.Input(

        shape=(224, 224, 3)
    )


    x = base_model(inputs)


    x = layers.GlobalAveragePooling2D(

        name="global_average_pooling2d"
    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Dropout(

        params["dropout"]
    )(x)


    x = layers.Dense(

        params["dense_units"],

        activation="relu"
    )(x)


    outputs = layers.Dense(

        4,

        activation="softmax"
    )(x)


    model = models.Model(

        inputs=inputs,

        outputs=outputs
    )


    return model


# ==============================================================================
# 9. MEMBANGUN MODEL SKENARIO 5
# ==============================================================================

def build_scenario_5():

    params = SCENARIO_5_PARAMS

    base_model = MobileNetV3Large(

        weights="imagenet",

        include_top=False,

        input_shape=(224, 224, 3)
    )


    base_model.trainable = True


    for layer in base_model.layers[
        :-params["fine_tune_layers"]
    ]:

        layer.trainable = False


    inputs = layers.Input(

        shape=(224, 224, 3)
    )


    x = base_model(inputs)


    x = layers.GlobalAveragePooling2D(

        name="global_average_pooling2d"
    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Dropout(

        params["dropout"]
    )(x)


    x = layers.Dense(

        params["dense_units"],

        activation="relu"
    )(x)


    outputs = layers.Dense(

        4,

        activation="softmax"
    )(x)


    model = models.Model(

        inputs=inputs,

        outputs=outputs
    )


    return model


# ==============================================================================
# 10. LOAD MODEL
# ==============================================================================
@st.cache_resource
    def load_phenotype_model(scenario):
    config = MODEL_CONFIG[scenario]
    model_path = config["model_path"]
    
    try:
        # ----------------------------------------------------------------------
        # SKENARIO 3 - INCEPTIONV3
        # ----------------------------------------------------------------------
        if scenario == 3:
            if not os.path.exists(model_path):
                with st.spinner("Mengunduh model InceptionV3..."):
                    urllib.request.urlretrieve(
                        MODEL_URL,
                        model_path
                    )
    
            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
    
        # ----------------------------------------------------------------------
        # SKENARIO 4 - EFFICIENTNETB0
        # ----------------------------------------------------------------------
        elif scenario == 4:
            if not os.path.exists(model_path):
                st.error(
                    f"File model tidak ditemukan:\n{model_path}"
                )
                return None
    
            model = build_scenario_4()
            model.load_weights(model_path)
    
        # ----------------------------------------------------------------------
        # SKENARIO 5 - MOBILENETV3LARGE
        # ----------------------------------------------------------------------
        elif scenario == 5:
            if not os.path.exists(model_path):
                st.error(
                    f"File model tidak ditemukan:\n{model_path}"
                )
                return None
    
            model = build_scenario_5()
            model.load_weights(model_path)
    
        else:
            st.error(
                f"Skenario {scenario} tidak didukung."
            )
            return None
    
        return model
    
    except Exception as e:
        st.error(
            f"Gagal memuat model Skenario {scenario}.\n\n"
            f"Error: {e}"
        )
        return None

# ==============================================================================
# 11. HEADER APLIKASI
# ==============================================================================

st.title(

    "🌶️ Sistem Penentuan Cluster Fenotipe Daun Cabai"
)


st.write(

    "Aplikasi berbasis *Deep Learning* untuk "
    "memetakan citra daun cabai ke dalam kelompok "
    "cluster fenotipe berdasarkan karakteristik "
    "representasi fitur yang telah dipelajari model."
)


# ==============================================================================
# 12. INFORMASI MODEL AKTIF
# ==============================================================================

model_name = config["name"]

input_size = config["input_size"]


st.subheader(

    f"Skenario {SELECTED_SCENARIO}: {model_name}"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(

        label="Model CNN",

        value=model_name
    )


with col2:

    st.metric(

        label="Ukuran Input",

        value=f"{input_size[0]}×{input_size[1]}"
    )


with col3:

    st.metric(

        label="Jumlah Cluster",

        value=config["num_classes"]
    )


st.divider()


# ==============================================================================
# 13. LOAD MODEL
# ==============================================================================

model = load_phenotype_model(

    SELECTED_SCENARIO
)


if model is not None:

    st.success(

        f"✅ Model {model_name} berhasil dimuat."
    )


# ==============================================================================
# 14. PILIH SUMBER GAMBAR
# ==============================================================================

st.subheader(

    "📷 Sumber Citra Daun Cabai"
)


input_method = st.radio(

    "Pilih metode input gambar:",

    [
        "Upload Gambar",
        "Ambil dari Kamera"
    ],

    horizontal=True
)


# ==============================================================================
# 15. UPLOAD GAMBAR
# ==============================================================================
uploaded_file = None
camera_image = None
if input_method == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Unggah citra daun cabai",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

# ==============================================================================
# 16. KAMERA
# ==============================================================================
elif input_method == "Ambil dari Kamera":
    camera_image = st.camera_input(
        "Ambil foto daun cabai menggunakan kamera"
    )

# ==============================================================================
# 17. MENENTUKAN GAMBAR
# ==============================================================================
image = None
if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    )
elif camera_image is not None:
    image = Image.open(
        camera_image
    )

# ==============================================================================
# 18. MENAMPILKAN GAMBAR
# ==============================================================================

if image is not None:

    image = image.convert("RGB")


    st.subheader(

        "🖼️ Citra Masukan"
    )


    st.image(
        image,
        caption="Citra Daun Cabai",
    )


    st.divider()


    # ==========================================================================
    # 19. INFORMASI PREPROCESSING
    # ==========================================================================

    st.subheader(

        "⚙️ Proses Analisis"
    )


    st.write(

        "1. Citra masukan dikonversi ke format RGB."
    )


    st.write(

        f"2. Citra diubah ukurannya menjadi "
        f"`{input_size[0]}×{input_size[1]}` piksel."
    )


    st.write(

        f"3. Nilai piksel diproses menggunakan "
        f"preprocessor {model_name}."
    )


    st.write(

        "4. Citra diproses oleh model CNN."
    )


    st.write(

        "5. Probabilitas setiap cluster dihitung."
    )


    # ==========================================================================
    # 20. TOMBOL PREDIKSI
    # ==========================================================================

    run_prediction = st.button(

        "🚀 Jalankan Pemetaan Cluster",

        type="primary",

        use_container_width=True
    )


    if run_prediction:

        if model is None:

            st.error(

                "Model belum berhasil dimuat."
            )


        else:

            with st.spinner(

                f"{model_name} sedang menganalisis "
                "citra daun cabai..."
            ):

                # ----------------------------------------------------------------
                # RESIZE
                # ----------------------------------------------------------------
                resized_image = image.resize(
                    input_size
                )

                # ----------------------------------------------------------------
                # ARRAY
                # ----------------------------------------------------------------
                img_array = np.array(
                    resized_image
                ).astype(
                    np.float32
                )

                # ----------------------------------------------------------------
                # BATCH DIMENSION
                # ----------------------------------------------------------------
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # ----------------------------------------------------------------
                # PREPROCESSING
                # ----------------------------------------------------------------
                img_array = config[
                    "preprocess"
                ](
                    img_array
                )

                # ----------------------------------------------------------------
                # PREDICTION
                # ----------------------------------------------------------------
                predictions = model.predict(
                    img_array,
                    verbose=0
                )
                highest_class_idx = int(
                    np.argmax(
                        predictions[0]
                    )
                )
                confidence_score = (
                    float(
                        predictions[0][
                            highest_class_idx
                        ]
                    ) * 100
                )
                predicted_cluster = (
                    CLASS_NAMES[
                        highest_class_idx
                    ]
                )


            # ==================================================================
            # 21. HASIL PREDIKSI
            # ==================================================================
            st.success(
                "✅ Pemetaan Cluster Berhasil!"
            )
            st.subheader(
                f"Hasil Prediksi: {predicted_cluster}"
            )

            # ==================================================================
            # 22. CONFIDENCE
            # ==================================================================
            st.progress(
                min(
                    int(confidence_score),
                    100
                )
            )
            st.write(
                f"📊 **Confidence Score:** "
                f"{confidence_score:.2f}%"
            )

            # ==================================================================
            # 23. DESKRIPSI CLUSTER
            # ==================================================================
            # st.info(

            #     f"💡 **Karakteristik Cluster**\n\n"
            #     f"{CLUSTER_DETAILS[predicted_cluster]}"
            # )

            # ==================================================================
            # 24. DISTRIBUSI PROBABILITAS
            # ==================================================================

            with st.expander(

                "📊 Lihat Distribusi Probabilitas "
                "Seluruh Cluster"
            ):

                for idx, class_name in enumerate(

                    CLASS_NAMES
                ):

                    probability = (

                        float(

                            predictions[0][idx]
                        ) * 100
                    )


                    st.write(

                        f"**{class_name}:** "
                        f"{probability:.2f}%"
                    )


else:

    st.info(

        "💡 Silakan upload gambar atau "
        "ambil foto menggunakan kamera untuk "
        "melakukan analisis cluster."
    )


# ==============================================================================
# 25. FOOTER
# ==============================================================================

st.divider()


st.caption(

    f"Model aktif: Skenario {SELECTED_SCENARIO} "
    f"({model_name})"
)
