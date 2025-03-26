// โหลดค่า aiSubtitleResult ที่เคยแสดงก่อนหน้านี้
// window.onload = function() {
//     let savedSubtitle = localStorage.getItem("savedSubtitle");
//     if (savedSubtitle) {
//         document.getElementById("aiSubtitleResult").innerText = savedSubtitle;
//     }
// };

// ป้องกันการรีเฟรชหน้าเว็บเมื่อกดปุ่ม
document.querySelector("button").addEventListener("click", function(event) {
    event.preventDefault(); 
});

let aiCategory = "ยังไม่มีข้อมูล";
let aiPrimaryCategory = "ยังไม่มีข้อมูล";
let aiRecommendation = "ยังไม่มีข้อมูล";

let libraryCategory = "ยังไม่มีข้อมูล";
let libraryPrimaryCategory = "ยังไม่มีข้อมูล";
let libraryRecommendation = "ยังไม่มีข้อมูล";

//ลบค่าเมื่อหน้าเว็บโหลดใหม่
window.onload = function() {
};

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predict-btn").addEventListener("click", function (event) {
        event.preventDefault();
    });
    
    // ตรวจจับการเปลี่ยนแท็บ
    document.getElementById("ai-tab").addEventListener("click", function () {
        updateCategoryAndRecommendation("ai");
    });

    document.getElementById("library-tab").addEventListener("click", function () {
        updateCategoryAndRecommendation("library");
    });
});

// ฟังก์ชันอัปเดตหมวดหมู่และคำแนะนำตามแท็บที่เลือก
function updateCategoryAndRecommendation(type) {
    const aiTab = document.getElementById("ai-tab");
    const libraryTab = document.getElementById("library-tab");
    if (type === "ai") {
        aiTab.classList.add("active");
        libraryTab.classList.remove("active");
        document.getElementById("categoryResult").innerText = aiCategory || "ไม่พบข้อมูล";
        document.getElementById("primarycategoryResult").innerText = aiPrimaryCategory || "ไม่พบข้อมูล";
        document.getElementById("recommendationResult").innerText = aiRecommendation || "ไม่มีคำแนะนำ";
    } else if (type === "library") {
        libraryTab.classList.add("active");
        aiTab.classList.remove("active");
        document.getElementById("categoryResult").innerText = libraryCategory || "ไม่พบข้อมูล";
        document.getElementById("primarycategoryResult").innerText = libraryPrimaryCategory || "ไม่พบข้อมูล";
        document.getElementById("recommendationResult").innerText = libraryRecommendation || "ไม่มีคำแนะนำ";
    }
}

async function downloadAndPredict() {

    let youtubeUrl = document.getElementById("youtubeUrl").value;

    if (!youtubeUrl) {
        alert("กรุณาใส่ลิงก์ YouTube");
        return;
    }

    let statusElement = document.getElementById("status");
    statusElement.innerText = "กำลังเริ่มดาวน์โหลด...";
    statusElement.style.color = "#e67e22";  

    // รีเซ็ตเครื่องเล่นเสียง
    let audioPlayer = document.getElementById("audioPlayer");
    let audioSource = document.getElementById("audioSource");

    document.getElementById("categoryResult").innerText = "กำลังอยู่ในขั้นตอนการคำนวน";
    document.getElementById("primarycategoryResult").innerText = "กำลังอยู่ในขั้นตอนการคำนวน";
    document.getElementById("recommendationResult").innerText = "กำลังอยู่ในขั้นตอนการคำนวน";
    document.getElementById("accuracyResult").innerText = "กำลังอยู่ในขั้นตอนการคำนวน";

    try {
        // เรียก API เพื่อดาวน์โหลด YouTube เป็น MP3
        let clearAudioResponse = await fetch("http://127.0.0.1:5000/clearAudio", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            mode: "cors"
        });
        let clearAudio = await clearAudioResponse.json();
        console.log("Clear Audio Response:", clearAudio);
        
        // แสดงแถบดาวน์โหลด (Progress Bar)
        let progressBar = document.createElement("progress");
        progressBar.value = 0;
        progressBar.max = 100;
        statusElement.appendChild(progressBar);

        let progressInterval = setInterval(() => {
            if (progressBar.value < 95) {
                progressBar.value += 5; // เพิ่ม Progress ทุกๆ 1 วิ
            }
        }, 1000);
        // เรียก API เพื่อดาวน์โหลด YouTube เป็น MP3
        let downloadResponse = await fetch("http://127.0.0.1:5000/download", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: youtubeUrl }),
            mode: "cors"
        });
        clearInterval(progressInterval);  // หยุดการเพิ่ม Progress
        progressBar.value = 100;  // ตั้งค่าเป็น 100% เมื่อเสร็จ

        let data = await downloadResponse.json();
        console.log("Download Response:", downloadResponse);
        if (data.success) {
            statusElement.innerText = "ดาวน์โหลดเสร็จแล้ว! กำลังพยากรณ์คำบรรยาย...";
            document.getElementById("aiSubtitleResult").innerText = "กำลังแปลงเสียงเป็นข้อความ...";
            document.getElementById("librarySubtitleResult").innerText = "กำลังแปลงเสียงเป็นข้อความ...";

            if (!audioPlayer.paused) {
                audioPlayer.pause();
            }
            // แสดงไฟล์เสียงที่ดาวน์โหลดมา
            let audioURL = `http://127.0.0.1:5000/static/${data.filename}`; 
            audioSource.src = audioURL;
            audioPlayer.load();

            // ตรวจสอบว่าไฟล์เสียงสามารถโหลดได้หรือไม่
            // let checkAudio = await fetch(audioURL, { method: "HEAD" });
            
            // if (checkAudio.ok) {
            //     audioSource.src = audioURL;
            //     audioPlayer.load();
            //     console.log("ไฟล์เสียง load สำเร็จ:", audioURL);
            // } else {
            //     console.log("ไฟล์เสียงไม่สามารถโหลดได้:", audioURL);
            // }

            // ส่งไปพยากรณ์คำบรรยาย
            let predictResponse = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ filename: data.filename }),
                mode: "cors"
            });

            let predictData = await predictResponse.json();
            console.log("Predict Response:", predictData);

            if (predictData.predictedSubtitle) {
                statusElement.innerText = "พยากรณ์คำบรรยายสำเร็จ!";
                document.getElementById("aiSubtitleResult").innerText = predictData.predictedSubtitle;
                document.getElementById("librarySubtitleResult").innerText = predictData.actualSubtitle;

                // แสดงค่า Accuracy
                document.getElementById("accuracyResult").innerText = `Accuracy: ${predictData.accuracy}%`;
                // ส่ง subtitle ไปจำแนกหมวดหมู่
                getRecommendation(predictData.predictedSubtitle, "ai");
                getRecommendation(predictData.actualSubtitle, "library");
            } else {
                statusElement.innerText = "เกิดข้อผิดพลาดในการพยากรณ์!";
                document.getElementById("aiSubtitleResult").innerText = "เกิดข้อผิดพลาดในการพยากรณ์!";
                document.getElementById("librarySubtitleResult").innerText = "เกิดข้อผิดพลาดในการพยากรณ์!";
            }
        } else {
            audioSource.src = "";
            audioPlayer.load(); // รีเซ็ต Player
            statusElement.innerText = "ไม่สามารถดาวน์โหลดวิดีโอ!";
            statusElement.style.color = "#e74c3c";
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("status").innerText = "เกิดข้อผิดพลาด!";
        document.getElementById("status").style.color = "#e74c3c";
    }
}

async function getRecommendation(subtitle, type) {
    let statusElement = document.getElementById("status");
    statusElement.innerText = "กำลังจำแนกหมวดหมู่และให้คำแนะนำ...";

    try {
        let response = await fetch("http://127.0.0.1:5000/recommendation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subtitle: subtitle}),
            mode: "cors"
        });

        let data = await response.json();
        console.log("Recommendation Response:", data);

        if (type === "ai") {
            aiCategory = data.top_3_categories || "ไม่พบข้อมูล";
            aiPrimaryCategory = data.primary_category || "ไม่พบข้อมูล";
            aiRecommendation = data.recommendation || "ไม่มีคำแนะนำ";
            statusElement.innerText = "จำแนกหมวดหมู่และให้คำแนะนำสำเร็จ!";
        } else if (type === "library") {
            libraryCategory = data.top_3_categories || "ไม่พบข้อมูล";
            libraryPrimaryCategory = data.primary_category || "ไม่พบข้อมูล";
            libraryRecommendation = data.recommendation || "ไม่มีคำแนะนำ";
            statusElement.innerText = "จำแนกหมวดหมู่และให้คำแนะนำสำเร็จ!";
        }

        // อัปเดตหมวดหมู่เริ่มต้น (AI เป็นค่าแรก)
        updateCategoryAndRecommendation("ai");

    } catch (error) {
        console.error("Error:", error);
        statusElement.innerText = "เกิดข้อผิดพลาดในการจำแนกหมวดหมู่!";
    }
}
