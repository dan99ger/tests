import cv2
import os
import time
import tempfile
import numpy as np
import streamlit as st

from PIL import Image
from ultralytics import YOLO


# =========================================================
# إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="AI Vision & Object Tracker",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 نظام الرؤية الحاسوبية وتتبع الكائنات")
st.caption("YOLOv8 Object Detection & Tracking باستخدام OpenCV وStreamlit")


# =========================================================
# تحميل نموذج YOLO
# =========================================================

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


model = load_yolo_model()


# =========================================================
# Session State
# =========================================================

if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = False


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("⚙️ إعدادات النظام")


option = st.sidebar.radio(
    "اختر نوع الوسائط:",
    [
        "صورة ثابتة (Image)",
        "مقطع فيديو (Video)"
    ]
)


confidence = st.sidebar.slider(
    "نسبة الثقة Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)


# =========================================================
# اختيار Tracker
# =========================================================

tracker_option = st.sidebar.selectbox(
    "خوارزمية التتبع:",
    [
        "ByteTrack",
        "BoT-SORT"
    ]
)


if tracker_option == "ByteTrack":
    tracker_file = "bytetrack.yaml"

else:
    tracker_file = "botsort.yaml"


# =========================================================
# الصورة
# =========================================================

if option == "صورة ثابتة (Image)":

    st.subheader("📷 اكتشاف الكائنات في الصور")

    uploaded_image = st.file_uploader(
        "قم برفع صورة:",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        img_array = np.array(image)


        # =================================================
        # تنفيذ الكشف
        # =================================================

        start_time = time.time()

        results = model(
            img_array,
            conf=confidence
        )

        processing_time = time.time() - start_time


        # =================================================
        # الصورة الناتجة
        # =================================================

        annotated_frame = results[0].plot()


        col1, col2 = st.columns(2)


        with col1:

            st.subheader("📷 الصورة الأصلية")

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.subheader("🎯 نتيجة الكشف")

            st.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )


        # =================================================
        # استخراج النتائج
        # =================================================

        detected_objects = []

        boxes = results[0].boxes


        if boxes is not None:

            for box in boxes:

                class_id = int(
                    box.cls[0]
                )

                score = float(
                    box.conf[0]
                )

                class_name = model.names[class_id]

                detected_objects.append(
                    {
                        "object": class_name,
                        "confidence": score
                    }
                )


        # =================================================
        # إحصائيات
        # =================================================

        st.divider()

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "عدد الكائنات",
                len(detected_objects)
            )


        with col2:

            unique_objects = len(
                set(
                    item["object"]
                    for item in detected_objects
                )
            )

            st.metric(
                "أنواع الكائنات",
                unique_objects
            )


        with col3:

            st.metric(
                "زمن المعالجة",
                f"{processing_time:.3f} ثانية"
            )


        # =================================================
        # عرض تفاصيل الكائنات
        # =================================================

        if detected_objects:

            st.subheader("🔍 تفاصيل الكائنات المكتشفة")


            for i, item in enumerate(
                detected_objects,
                start=1
            ):

                st.write(
                    f"**{i}. {item['object']}** "
                    f"— الثقة: "
                    f"{item['confidence'] * 100:.2f}%"
                )

        else:

            st.warning(
                "لم يتم اكتشاف أي كائن."
            )


# =========================================================
# الفيديو
# =========================================================

elif option == "مقطع فيديو (Video)":

    st.subheader("🎥 تتبع الكائنات في الفيديو")


    uploaded_video = st.file_uploader(
        "قم برفع مقطع فيديو:",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded_video:

        # =================================================
        # حفظ الفيديو مؤقتاً
        # =================================================

        suffix = os.path.splitext(
            uploaded_video.name
        )[1]


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_video:

            temp_video.write(
                uploaded_video.read()
            )

            video_path = temp_video.name


        # =================================================
        # فتح الفيديو
        # =================================================

        cap = cv2.VideoCapture(
            video_path
        )


        # معلومات الفيديو
        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        original_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        # =================================================
        # معلومات الفيديو
        # =================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "عدد Frames",
                total_frames
            )


        with col2:

            st.metric(
                "FPS الأصلي",
                f"{original_fps:.1f}"
            )


        with col3:

            st.metric(
                "العرض",
                width
            )


        with col4:

            st.metric(
                "الارتفاع",
                height
            )


        # =================================================
        # أزرار التحكم
        # =================================================

        col_start, col_stop = st.columns(2)


        with col_start:

            start_tracking = st.button(
                "▶️ بدء التتبع",
                type="primary"
            )


        with col_stop:

            if st.button("⏹️ إيقاف التتبع"):

                st.session_state.stop_tracking = True


        # =================================================
        # تشغيل الفيديو
        # =================================================

        if start_tracking:

            st.session_state.stop_tracking = False


            frame_placeholder = st.empty()

            statistics_placeholder = st.empty()

            progress_bar = st.progress(0)


            frame_number = 0

            unique_track_ids = set()


            while cap.isOpened():

                if st.session_state.stop_tracking:

                    break


                ret, frame = cap.read()


                if not ret:

                    break


                frame_number += 1


                # =========================================
                # حساب الزمن
                # =========================================

                start_time = time.time()


                # =========================================
                # YOLO Tracking
                # =========================================

                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_file,
                    conf=confidence,
                    verbose=False
                )


                processing_time = (
                    time.time() - start_time
                )


                # =========================================
                # FPS
                # =========================================

                current_fps = (
                    1 / processing_time
                    if processing_time > 0
                    else 0
                )


                # =========================================
                # استخراج Track IDs
                # =========================================

                current_ids = []

                boxes = results[0].boxes


                if (
                    boxes is not None
                    and boxes.id is not None
                ):

                    current_ids = (
                        boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    unique_track_ids.update(
                        current_ids
                    )


                # =========================================
                # رسم النتائج
                # =========================================

                annotated_frame = (
                    results[0].plot()
                )


                # OpenCV BGR -> RGB
                annotated_frame = (
                    cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )
                )


                # =========================================
                # عرض Frame
                # =========================================

                frame_placeholder.image(
                    annotated_frame,
                    channels="RGB",
                    use_container_width=True
                )


                # =========================================
                # الإحصائيات
                # =========================================

                number_objects = (
                    len(boxes)
                    if boxes is not None
                    else 0
                )


                statistics_placeholder.markdown(
                    f"""
### 📊 معلومات التتبع

- **Frame:** {frame_number} / {total_frames}
- **الكائنات في Frame الحالي:** {number_objects}
- **إجمالي Track IDs المختلفة:** {len(unique_track_ids)}
- **سرعة المعالجة:** {current_fps:.2f} FPS
- **Tracker:** {tracker_option}
                    """
                )


                # =========================================
                # Progress Bar
                # =========================================

                if total_frames > 0:

                    progress = min(
                        frame_number / total_frames,
                        1.0
                    )

                    progress_bar.progress(
                        progress
                    )


            cap.release()


            if not st.session_state.stop_tracking:

                progress_bar.progress(1.0)

                st.success(
                    "✅ انتهت عملية تتبع الفيديو."
                )

            else:

                st.warning(
                    "⏹️ تم إيقاف عملية التتبع."
                )


        else:

            cap.release()


    else:

        st.info(
            "👆 قم برفع فيديو لبدء عملية التتبع."
        )