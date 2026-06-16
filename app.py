import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="GamingMind Intelligent Screening Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styles
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #2E4053; margin-bottom: 5px; }
    .subtitle { font-size: 16px; color: #7F8C8D; margin-bottom: 25px; }
    .section-header { font-size: 22px; font-weight: bold; color: #1F618D; margin-top: 15px; margin-bottom: 15px; }
    
    /* Card Styles */
    .card-risk-high { background-color: #FDEDEC; padding: 25px; border-radius: 10px; border-left: 8px solid #E74C3C; margin-bottom: 20px; color: #1C2833; }
    .card-risk-med { background-color: #FEF9E7; padding: 25px; border-radius: 10px; border-left: 8px solid #F1C40F; margin-bottom: 20px; color: #1C2833; }
    .card-risk-low { background-color: #EAF2F8; padding: 25px; border-radius: 10px; border-left: 8px solid #2ECC71; margin-bottom: 20px; color: #1C2833; }
    .card-intervention { background-color: #F4F9F4; padding: 25px; border-radius: 10px; border-left: 8px solid #2ECC71; margin-bottom: 20px; color: #2C3E50; }
    
    /* Scale Box Styles */
    .scale-box-left { background-color: #FDEDEC; padding: 20px; border-radius: 12px; border: 2px solid #F5B7B1; min-height: 220px; color: #2C3E50; }
    .scale-box-right { background-color: #EAF2F8; padding: 20px; border-radius: 12px; border: 2px solid #AED6F1; min-height: 220px; color: #2C3E50; }
    .scale-status { text-align: center; font-size: 18px; font-weight: bold; margin: 20px 0; padding: 12px; border-radius: 8px; }
    
    /* Healing & Medical Page Styles */
    .healing-card { background-color: #F9EBEA; padding: 20px; border-radius: 12px; border: 1.5px solid #F2D7D5; margin-bottom: 15px; color: #2C3E50; }
    .game-recommend-card { background-color: #EBF5FB; padding: 20px; border-radius: 12px; border: 1.5px solid #AED6F1; margin-bottom: 15px; color: #2C3E50; }
    .real-world-card { background-color: #E8F8F5; padding: 20px; border-radius: 12px; border: 1.5px solid #A9DFBF; margin-bottom: 15px; color: #2C3E50; }
    
    /* Icons */
    .icon-warning { color: #E74C3C; font-weight: bold; margin-right: 8px; }
    .icon-success { color: #2ECC71; font-weight: bold; margin-right: 8px; }
    
    /* 🌐 Sleek Option Buttons */
    div.stButton > button {
        text-align: left !important;
        padding-left: 20px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    
    /* 👤 Identity Descriptive Cards */
    .identity-card-player {
        width: 100%;                  /* 强制撑满左边列宽 */
        box-sizing: border-box;       /* 防止 padding 撑破布局 */
        height: 260px;                /* 强制固定高度，确保两边绝对等高 */
        display: flex;
        flex-direction: column;
        justify-content: center;      /* 内容垂直居中 */
        align-items: center;          /* 内容水平居中 */
        
        background-color: #EAF2F8;
        border: 2px solid #AED6F1;
        padding: 25px;
        border-radius: 15px;
        color: #2C3E50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .identity-card-medical {
        width: 100%;                  /* 强制撑满右边列宽 */
        box-sizing: border-box;       /* 防止 padding 撑破布局 */
        height: 260px;                /* 强制固定高度，确保两边绝对等高 */
        display: flex;
        flex-direction: column;
        justify-content: center;      /* 内容垂直居中 */
        align-items: center;          /* 内容水平居中 */
        
        background-color: #E8F8F5;
        border: 2px solid #A9DFBF;
        padding: 25px;
        border-radius: 15px;
        color: #2C3E50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .portal-icon { font-size: 55px; margin-bottom: 15px; }
    .portal-title { font-size: 24px; font-weight: bold; margin-bottom: 12px; }
    .portal-desc { font-size: 14px; color: #5D6D7E; line-height: 1.5; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Multi-Language Translation Dictionary (I18n)
# ==========================================
LOCALES = {
    "zh": {
        "welcome_title": "🧠 GamingMind 智能筛查平台",
        "welcome_sub": "欢迎使用！请选择您的语言和身份以开启个性化评估。",
        "portal_lang_title": "第一步：选择您的语言",
        "portal_id_title": "第二步：选择您的身份并进入系统",
        "btn_back": "↩️ 切换身份/语言",
        "player": "玩家 / 家长 (科普自评版)",
        "medical": "医务人员 (临床决策辅助版)",
        "panel_title": "快速评估面板",
        "report_title": "评估报告",
        "q_age_group": "1. 选择年龄段",
        "q_age_exact": "└─ 选择具体年龄",
        "q_gender": "2. 性别",
        "q_hours": "3. 每周游戏时长 (小时/周)",
        "q_game_type": "4. 主要游戏类型",
        "q_behavior": "5. 实际游玩与互动行为 (可多选 - 点击切换状态)",
        "behavior_solo": "独自游玩 (孤狼玩家)",
        "behavior_social": "积极社交 (在社交媒体/论坛分享、交流)",
        "btn_eval": "开始评估",
        "evaluating": "正在分析行为习惯并绘制多维图谱...",
        "info_prompt": "请在左侧输入信息并点击“开始评估”生成报告。",
        
        # Identity Cards (zh)
        "card_play_title": "玩家 / 家长",
        "card_play_desc": "适合个人自测与家庭科普。提供通俗易懂的“心理天平”分析及日常解压小妙招。",
        "card_play_btn": "以 玩家/家长 身份进入",
        "card_med_title": "医务人员",
        "card_med_desc": "适合临床辅助与学术研究。提供多维度量化风险图表及循证医学干预指南。",
        "card_med_btn": "以 医务人员 身份进入",
        
        # Risk Levels (Player)
        "risk_high_title": "⚠️ 高焦虑风险",
        "risk_high_desc": "当前的娱乐与生活平衡面临较大压力。高强度的游戏时长与成长阶段的压力产生了叠加效应。建议适当调整，防范情绪倦怠。",
        "risk_med_title": "🟡 中度焦虑风险",
        "risk_med_desc": "您目前处于“黄灯”预警区间。虽然未达临界，但游戏习惯已开始对日常生活产生轻微代偿，微调生活作息即可轻松恢复健康状态。",
        "risk_low_title": "🟢 低焦虑风险",
        "risk_low_desc": "太棒了！游戏充分发挥了它解压、放松的积极作用。您的生活与娱乐平衡得非常好，请继续保持！",
        
        # Risk Levels (Medical)
        "cli_high_title": "⚠️ 临床高风险预警 (High Clinical Risk)",
        "cli_high_desc": "受试者的行为特征与高焦虑表征高度共现。建议结合 Hamilton 焦虑量表 (HAMA) 进行深度临床访谈，并评估是否存在游戏障碍 (GD) 倾向。",
        "cli_med_title": "🟡 临床中度风险预警 (Moderate Clinical Risk)",
        "cli_med_desc": "受试者呈现亚临床焦虑状态。建议进行预防性认知行为干预 (CBT)，并监测其睡眠质量与线下社会功能恢复情况。",
        "cli_low_title": "🟢 临床低风险状态 (Low Clinical Risk)",
        "cli_low_desc": "受试者各项筛查指标均在健康基线范围内。无需临床干预，建议维持常规随访。",
        
        # Feature 2 (Scale & Chart)
        "scale_title": "您的心理健康天平",
        "scale_sub": "我们将您的生活习惯视为一个天平。左侧增加压力，右侧提供保护：",
        "plate_left": " 左盘 (压力砝码)",
        "plate_right": " 右盘 (保护盾牌)",
        "scale_left_tilt": "天平状态：向左倾斜 ( 压力砝码过重，建议适当调整)",
        "scale_right_tilt": "天平状态：向右倾斜 ( 保护伞充足，心理韧性良好)",
        "scale_balance": "天平状态：完美平衡 ( 处于临界平衡状态)",
        
        "cli_chart_title": " 多维心理健康临床图谱 (Clinical Radar Chart)",
        "cli_chart_sub": "基于受试者动态数据的五维病理特征偏离度分析（对比健康对照组）：",
        
        # Protectors
        "prot_self": "<b>主动自省 (Self-Awareness)</b>: 您正在主动评估自己的生活状态，这是心理健康最强大的第一道防线！",
        
        # Feature 3 (New Page for Player & Medical)
        "heal_title": "专属解压与心灵疗愈空间",
        "heal_sub": "游戏是极好的避风港，但现实世界同样温暖。我们为您量身定制了以下疗愈方案：",
        "btn_next_page": "进入下一页：开启您的解压疗愈之旅 ",
        "btn_next_page_med": " 进入下一页：查看临床干预与健康指导医嘱 ",
        "btn_back_report": "↩️ 返回评估报告",
        "btn_back_report_med": "↩️ 返回临床评估报告",
        "heal_box_prompt": "评估完成！我们为您准备了一份「专属解压与心灵疗愈指南」。",
        "med_box_prompt": "评估完成！系统已生成「专属临床干预与健康指导医嘱」。",
        
        # 🩺 Medical Feature 3 Advice (zh)
        "med_advice_title": "临床医学干预与健康指导建议",
        "med_advice_sub": "根据受试者的年龄段及生理、行为筛查特征，提供以下针对性的临床医学干预方案：",
        "med_u18_guardian_title": "给监护人/家长的医学建议 (Advice for Guardians)",
        "med_u18_patient_title": "给青少年本人的生理健康建议 (Advice for Adolescent Patient)",
        "med_o18_patient_title": "给成年患者本人的生理与心理健康建议 (Advice for Adult Patient)"
    },
    "en": {
        "welcome_title": "🧠 GamingMind Screening Platform",
        "welcome_sub": "Welcome! Please select your language and identity to start the personalized assessment.",
        "portal_lang_title": "Step 1: Select Your Language",
        "portal_id_title": "Step 2: Choose Your Identity to Enter",
        "btn_back": "↩️ Switch Identity/Language",
        "player": "Player / Parent (Self-Service Edition)",
        "medical": "Medical Professional (Clinical Support Edition)",
        "panel_title": "Quick Assessment Panel",
        "report_title": "Your Personalized Report",
        "q_age_group": "1. Select Age Group",
        "q_age_exact": "└─ Select Exact Age",
        "q_gender": "2. Gender",
        "q_hours": "3. Weekly Gaming Hours",
        "q_game_type": "4. Primary Game Type",
        "q_behavior": "5. Actual Social Behavior (Multi-select - Click to toggle)",
        "behavior_solo": "Solo Play (Lone Wolf)",
        "behavior_social": "Active Socializing (Sharing on Forums/Social Media)",
        "btn_eval": "Get Results & Advice",
        "evaluating": "Analyzing habits and plotting clinical radar chart...",
        "info_prompt": "Please answer the questions on the left and click 'Get Results' to see your report.",
        
        # Identity Cards (en)
        "card_play_title": "Player / Parent",
        "card_play_desc": "Ideal for self-assessment and family education. Features an intuitive 'Balance Scale' and daily stress-relief tips.",
        "card_play_btn": "Enter as Player / Parent",
        "card_med_title": "Medical Professional",
        "card_med_desc": "Designed for clinical decision support. Features quantitative risk charts and evidence-based protocols.",
        "card_med_btn": "Enter as Medical Professional",
        
        # Risk Levels (Player)
        "risk_high_title": "High Anxiety Risk",
        "risk_high_desc": "Your current lifestyle balance is heavily strained. High gaming hours combined with developmental age pressures are likely acting as stressors. Action is highly recommended.",
        "risk_med_title": "Moderate Anxiety Risk",
        "risk_med_desc": "You are in the 'yellow light' warning zone. While not critical, gaming habits are starting to overlap with stress factors. Minor adjustments will easily bring you back.",
        "risk_low_title": "Low Anxiety Risk",
        "risk_low_desc": "Great job! Gaming is serving its true purpose—a fun, healthy way to unwind. There are no signs of gaming-related anxiety.",
        
        # Risk Levels (Medical)
        "cli_high_title": "High Clinical Risk Warning",
        "cli_high_desc": "The patient's behavioral profile aligns strongly with high-anxiety cohorts. Clinical interview paired with HAMA scale is recommended to evaluate potential Gaming Disorder (GD).",
        "cli_med_title": "Moderate Clinical Risk Warning",
        "cli_med_desc": "The patient presents subclinical anxiety. Preventive Cognitive Behavioral Therapy (CBT) and sleep hygiene monitoring are advised.",
        "cli_low_title": "Low Clinical Risk Status",
        "cli_low_desc": "All screening indicators are within healthy baseline ranges. No clinical intervention is required at this stage.",
        
        # Feature 2 (Scale & Chart)
        "scale_title": " Feature 2: Your Mental Health Balance Scale",
        "scale_sub": "We analyze your lifestyle like a scale. Left side adds stress, right side keeps you balanced:",
        "plate_left": " Left Plate (Stress Weights)",
        "plate_right": " Right Plate (Protective Shields)",
        "scale_left_tilt": "Scale Status: Tilting Left ( Stress heavy! Needs attention)",
        "scale_right_tilt": "Scale Status: Tilting Right ( Well protected! Keep it up)",
        "scale_balance": "Scale Status: Balanced ( Borderline equilibrium)",
        
        "cli_chart_title": "Feature 2: Multi-dimensional Clinical Radar Chart",
        "cli_chart_sub": "Five-dimensional pathological deviation analysis compared with healthy baseline:",
        
        # Protectors
        "prot_self": "<b>Proactive Self-Awareness</b>: By taking this screening, you are actively reflecting on your lifestyle. This is the first step to wellness!",
        
        # Feature 3 (New Page for Player & Medical)
        "heal_title": "Your Healing & Stress-Relief Space",
        "heal_sub": "Gaming is a great harbor, but the real world is just as warm. Explore our curated healing guide:",
        "btn_next_page": "Next Page: Start Your Healing Journey ",
        "btn_next_page_med": "Next Page: View Clinical Intervention Protocols ",
        "btn_back_report": "↩️ Back to Assessment Report",
        "btn_back_report_med": "↩️ Back to Clinical Report",
        "heal_box_prompt": "Evaluation complete! We have prepared a 'Personalized Healing & Stress-Relief Guide' for you.",
        "med_box_prompt": "Evaluation complete! We have generated 'Personalized Clinical Intervention & Health Advice' for you.",
        
        # 🩺 Medical Feature 3 Advice (en)
        "med_advice_title": "Clinical Medical Intervention & Health Advice",
        "med_advice_sub": "Based on the subject's age group, physiological, and behavioral screening profile, the following targeted clinical protocols are recommended:",
        "med_u18_guardian_title": "Medical Advice for Guardians/Parents",
        "med_u18_patient_title": "Physical Health Advice for the Adolescent Patient",
        "med_o18_patient_title": "Physical & Mental Health Advice for the Adult Patient"
    }
}

# ==========================================
# 3. Session State Routing & Component Logic
# ==========================================
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "identity" not in st.session_state:
    st.session_state.identity = "Player"
if "language" not in st.session_state:
    st.session_state.language = "zh"
# Initialize behavior states
if "selected_behaviors" not in st.session_state:
    st.session_state.selected_behaviors = ["solo"]
# Initialize sub-page routing for both Player and Medical
if "player_subpage" not in st.session_state:
    st.session_state.player_subpage = "main"
if "medical_subpage" not in st.session_state:
    st.session_state.medical_subpage = "main"

# ==========================================
# 🚪 ROUTE 1: Landing Page
# ==========================================
if not st.session_state.confirmed:
    st.markdown('<div class="main-title" style="text-align:center; margin-top:40px;">🧠 GamingMind Intelligent Screening Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="text-align:center; margin-bottom: 30px;">欢迎使用智能心理健康筛查系统 / Welcome to the Screening Portal</div>', unsafe_allow_html=True)
    
    # Center container
    _, center_col, _ = st.columns([0.8, 2, 0.8])
    with center_col:
        
        # 🌐 STEP 1: Slim Language Cards
        st.write(f"##### {LOCALES[st.session_state.language]['portal_lang_title']}")
        col_lang_zh, col_lang_en = st.columns(2)
        
        with col_lang_zh:
            is_active = st.session_state.language == "zh"
            zh_label = "🔴 🇨🇳 中文 (Chinese)" if is_active else "🇨🇳 中文 (Chinese)"
            if st.button(zh_label, key="btn_lang_zh", use_container_width=True):
                st.session_state.language = "zh"
                st.rerun()
                
        with col_lang_en:
            is_active = st.session_state.language == "en"
            en_label = "🔴 🇺🇸 English" if is_active else "🇺🇸 English"
            if st.button(en_label, key="btn_lang_en", use_container_width=True):
                st.session_state.language = "en"
                st.rerun()
        
        st.write("---")
        t_portal = LOCALES[st.session_state.language]
        
                # 👤 STEP 2: Identity Descriptive Cards & Buttons
        st.write(f"##### {t_portal['portal_id_title']}")
        col_play, col_med = st.columns(2, gap="large")
        
        with col_play:
            st.markdown(f"""
            <div class="identity-card-player">
                <div class="portal-icon">🎮</div>
                <div class="portal-title">{t_portal["card_play_title"]}</div>
                <div class="portal-desc">{t_portal["card_play_desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button(t_portal["card_play_btn"], key="btn_enter_player", use_container_width=True, type="primary"):
                # --- 清空历史评估缓存，确保白纸进入 ---
                for key in ["last_risk_prob", "last_drivers", "last_protectors", "last_hours", "last_age", "last_gender", "last_has_solo", "last_has_social"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state.identity = "Player"
                st.session_state.confirmed = True
                st.session_state.player_subpage = "main"
                st.rerun()
                
        with col_med:
            st.markdown(f"""
            <div class="identity-card-medical">
                <div class="portal-icon">🩺</div>
                <div class="portal-title">{t_portal["card_med_title"]}</div>
                <div class="portal-desc">{t_portal["card_med_desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button(t_portal["card_med_btn"], key="btn_enter_medical", use_container_width=True, type="primary"):
                # --- 清空历史评估缓存，确保白纸进入 ---
                for key in ["last_risk_prob", "last_drivers", "last_protectors", "last_hours", "last_age", "last_gender", "last_has_solo", "last_has_social"]:
                    if key in st.session_state:
                        del st.session_state[key]
                        
                st.session_state.identity = "Medical"
                st.session_state.confirmed = True
                st.session_state.medical_subpage = "main"
                st.rerun()


# ==========================================
# 🖥️ ROUTE 2: Main Application Dashboard (Start)
# ==========================================
else:
    lang = st.session_state.language
    identity = st.session_state.identity
    t = LOCALES[lang]
    
        # Top Bar with Reset Button
    top_col_title, top_col_btn = st.columns([4, 1])
    with top_col_title:
        st.markdown(f'<div class="main-title">{t["welcome_title"]} <span style="font-size:16px; color:#3498DB;">({t["player"] if identity == "Player" else t["medical"]})</span></div>', unsafe_allow_html=True)
    with top_col_btn:
        if st.button(t["btn_back"], use_container_width=True):
            # --- 返回主舱时清空历史评估数据 ---
            for key in ["last_risk_prob", "last_drivers", "last_protectors", "last_hours", "last_age", "last_gender", "last_has_solo", "last_has_social"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.confirmed = False
            st.rerun()

            
    st.write("---")

        # =========================================================================
    # PLAYER - SUBPAGE 2: Healing & Recommendations Page (卡片悬停平滑展开版)
    # =========================================================================
    if identity == "Player" and st.session_state.player_subpage == "recommendations":
        # 注入局部 CSS 样式，实现卡片悬停时平滑展开详细信息，并确保两列卡片完美等宽、对齐
        st.markdown("""
            <style>
            /* 🎮 游戏推荐卡片 - 基础样式 */
            .game-recommend-card {
                width: 100%;
                box-sizing: border-box;
                height: 160px; /* 统一初始高度 */
                margin-bottom: 20px;
                padding: 24px;
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-left: 5px solid #2980B9; /* 蓝色左边条 */
                border-radius: 12px;
                transition: height 0.4s ease, transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
                overflow: hidden; /* 隐藏折叠内容 */
                cursor: pointer;
            }

            /* 🎮 游戏推荐卡片 - 悬停展开样式 */
            .game-recommend-card:hover {
                height: 320px; /* 悬停展开后的高度 */
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(41, 128, 185, 0.1), 0 4px 6px -2px rgba(41, 128, 185, 0.05);
                background-color: #F0F9FF; /* 悬停时变为淡蓝色 */
                border-color: #BAE6FD;
            }

            /* 悬停详细信息区域默认隐藏 */
            .hover-details {
                max-height: 0px;
                overflow: hidden;
                transition: max-height 0.4s ease, margin-top 0.4s ease, padding-top 0.4s ease;
                margin-top: 0px;
                padding-top: 0px;
                border-top: 0px dashed transparent;
                font-size: 14px;
                color: #334155;
                line-height: 1.6;
            }

            /* 鼠标悬停在卡片上时，平滑展开并显示蓝色虚线分割线 */
            .game-recommend-card:hover .hover-details {
                max-height: 200px; /* 足够容纳文字的高度 */
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1.5px dashed #93C5FD;
            }

            /* 🌿 现实世界充电站卡片 - 基础样式 */
            .real-world-card {
                width: 100%;
                box-sizing: border-box;
                height: 160px; /* 与游戏卡片初始高度完全一致，确保完美对齐 */
                margin-bottom: 20px;
                padding: 24px;
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-left: 5px solid #27AE60; /* 绿色左边条 */
                border-radius: 12px;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
                display: flex;
                flex-direction: column;
                justify-content: center; /* 内容垂直居中 */
            }

            /* 🌿 现实世界充电站卡片 - 悬停微动 */
            .real-world-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(39, 174, 96, 0.1), 0 4px 6px -2px rgba(39, 174, 96, 0.05);
                background-color: #F0FDF4; /* 悬停时变为淡绿色 */
                border-color: #BBF7D0;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="main-title" style="color: #27AE60;">{t["heal_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">{t["heal_sub"]}</div>', unsafe_allow_html=True)
        
        if st.button(t["btn_back_report"], key="btn_back_top", type="secondary"):
            st.session_state.player_subpage = "main"
            st.rerun()
            
        st.write("")
        col_rec_left, col_rec_right = st.columns(2, gap="large")
        
        with col_rec_left:
            if lang == "en":
                st.markdown("### Therapeutic Games")
                st.write("Hover your mouse over the cards below to reveal the story background and gameplay rules:")
                
                st.markdown("""
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">It Takes Two</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>Mandatory two-player cooperation. Engaging in lighthearted communication heavily dilutes loneliness and rebuilds interpersonal connections.</b>
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>Story Background:</b> A broken couple is turned into clay dolls by a magic book. They must work together to repair their relationship and return to the real world.</p>
                        <p style="margin: 0;"><b>Gameplay Rules:</b> A strict two-player cooperative game. Two players must cooperate using completely different, unique abilities in each level to solve puzzles and pass challenges.</p>
                    </div>
                </div>
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">Stardew Valley</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>Free, slow-paced pixel pastoral life. There are no deadlines here, allowing tense nerves to fully relax.</b> 
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>Story Background:</b> You inherit your grandfather's old farm plot in Stardew Valley. Armed with hand-me-down tools and a few coins, you set out to begin your new life.</p>
                        <p style="margin: 0;"><b>Gameplay Rules:</b> A simulation role-playing game. Players raise livestock, grow crops, mine ores, socialise with townspeople, and rebuild the community at their own pace.</p>
                    </div>
                </div>
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">Sky: Children of the Light</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>A visual feast of soaring through the clouds. It promotes wordless kindness and pure mutual assistance, awakening inner peace.</b>
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>Story Background:</b> As the Children of the Light, players spread hope through a desolate kingdom to return fallen Stars to their constellations.</p>
                        <p style="margin: 0;"><b>Gameplay Rules:</b> An open-world social adventure game. Players fly across beautiful realms, solve puzzles together, and light candles to connect with other players without verbal communication.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 疗愈系游戏推荐 (Therapeutic Games)")
                st.write("将鼠标悬停在下方卡片上，即可展开查看游戏的故事背景与玩法规则：")
                
                st.markdown("""
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">《双人成行》 (It Takes Two)</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>双人合作。在欢声笑语的沟通中，能极大地稀释孤独感，重建人际连接。</b> 
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>故事背景：</b> 一对感情破裂的夫妻被魔法书变成了泥人玩偶，他们必须齐心协力通过重重考验，修复彼此的关系并回到现实世界。</p>
                        <p style="margin: 0;"><b>玩法规则：</b> 严格的双人合作游戏。两位玩家必须使用各自完全不同且独特的关卡能力，默契配合进行解谜、战斗与跑酷。</p>
                    </div>
                </div>
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">《星露谷物语》 (Stardew Valley)</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>自由的慢节奏像素田园生活。在这里没有ddl，能让紧绷的神经彻底松弛。</b> 
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>故事背景：</b> 你继承了爷爷在星露谷留下的旧农场。带着几件简旧的农具和几枚硬币，你决定离开喧嚣的都市，开启全新的田园生活。</p>
                        <p style="margin: 0;"><b>玩法规则：</b> 模拟经营与角色扮演游戏。玩家可以自由决定每日行程，包括开垦荒地、种植作物、饲养动物、去矿洞冒险，以及与小镇居民建立友谊。</p>
                    </div>
                </div>
                <div class="game-recommend-card">
                    <h4 style="color: #2980B9; margin: 0 0 8px 0;">《光·遇》 (Sky: Children of the Light)</h4>
                    <p style="margin-bottom:0; font-size:14px; line-height:1.5;">
                        <b>翱翔云端的视觉盛宴，推崇无言的善意与纯粹的互助，唤醒内心的宁静。</b> 
                    </p>
                    <div class="hover-details">
                        <p style="margin: 0 0 8px 0;"><b>故事背景：</b> 玩家作为“光之后裔”，肩负着将光明与星星带回失落王国的使命，在云端与荒野间展开寻找自我的旅程。</p>
                        <p style="margin: 0;"><b>玩法规则：</b> 开放世界社交冒险游戏。玩家在美丽的场景中翱翔，通过点亮蜡烛与其他玩家相识，无需言语，仅靠动作、琴声和牵手共同解谜通关。</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_rec_right:
            if lang == "en":
                st.markdown("### Real-World Recharge")
                st.write("Besides gaming, there are many clinically proven healthy activities in the physical world that quickly lower stress hormones:")
                
                st.markdown("""
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">15-Minute "Green Exercise"</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>Scientific Principle:</b> Walk in a park for 15 minutes. Bilateral alternating movement during walking directly reduces amygdala activity, naturally burning away stress.
                    </p>
                </div>
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">Offline "Face-to-Face" Socializing</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>Scientific Principle:</b> Meet a friend offline. Real-time facial interaction stimulates oxytocin secretion, acting as a natural antidote to anxiety.
                    </p>
                </div>
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">Active Exercise & Stretching</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>Scientific Principle:</b> 30 minutes of jogging or yoga. Moderate physical fatigue prompts the brain to secrete endorphins, bringing physical pleasure and improving sleep.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 现实世界充电站 (Real-World Recharge)")
                st.write("除了游戏，现实世界中也有许多经过临床证实的、能快速降低压力荷尔蒙的健康消遣：")
                
                st.markdown("""
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">15分钟“绿色森林浴” (Green Exercise)</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>科学原理：</b> 去公园散步15分钟。步行时的双侧交替运动能直接降低大脑杏仁核的活跃度，自然烧掉压力。
                    </p>
                </div>
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">线下“面对面”茶话会 (Real Connection)</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>科学原理：</b> 约一位好友线下聚会。面部表情的实时互动能刺激催产素的分泌，是天然的焦虑解药。
                    </p>
                </div>
                <div class="real-world-card">
                    <h4 style="color: #27AE60; margin: 0 0 8px 0;">暴汗运动与拉伸 (Endorphin Release)</h4>
                    <p style="margin:0; font-size:14px; line-height:1.5;">
                        <b>科学原理：</b> 30分钟慢跑或瑜伽。身体的适度疲劳会促使大脑分泌内啡肽，带来生理愉悦并改善睡眠。
                    </p>
                </div>
                """, unsafe_allow_html=True)

            
        st.write("---")
        if st.button(t["btn_back_report"], key="btn_back_bottom", type="secondary", use_container_width=True):
            st.session_state.player_subpage = "main"
            st.rerun()




    # =========================================================================
    # MEDICAL - SUBPAGE 2: Clinical Intervention Page (医生端动态临床医嘱页面 - 双语无 Emoji 版)
    # =========================================================================
    elif identity == "Medical" and st.session_state.medical_subpage == "intervention":
        # 获取受试者的年龄和计算出的风险概率
        age_val = st.session_state.get("last_age", 22)
        risk_prob = st.session_state.get("last_risk_prob", 0.35)
        
        # 根据风险概率划定临床等级 (支持双语)
        if risk_prob >= 0.70:
            risk_level = "High"
            risk_color = "#C0392B"
            risk_bg = "#FDEDEC"
            if lang == "en":
                risk_label = "High Risk - Clinical Intervention Required"
                status_banner_text = f"Current Subject Status: {risk_label} (Score: {risk_prob*100:.0f}%)"
            else:
                risk_label = "高风险 (High Risk) - 需临床医学干预"
                status_banner_text = f"当前受试者状态：{risk_label} (得分: {risk_prob*100:.0f}%)"
        elif risk_prob >= 0.40:
            risk_level = "Moderate"
            risk_color = "#D4AC0D"
            risk_bg = "#FEF9E7"
            if lang == "en":
                risk_label = "Moderate Risk - Behavioral Guidance & Prevention"
                status_banner_text = f"Current Subject Status: {risk_label} (Score: {risk_prob*100:.0f}%)"
            else:
                risk_label = "中度风险 (Moderate Risk) - 需行为引导与预防"
                status_banner_text = f"当前受试者状态：{risk_label} (得分: {risk_prob*100:.0f}%)"
        else:
            risk_level = "Low"
            risk_color = "#27AE60"
            risk_bg = "#EAF2F8"
            if lang == "en":
                risk_label = "Low Risk - Routine Health Promotion"
                status_banner_text = f"Current Subject Status: {risk_label} (Score: {risk_prob*100:.0f}%)"
            else:
                risk_label = "低风险 (Low Risk) - 维持常规健康促进"
                status_banner_text = f"当前受试者状态：{risk_label} (得分: {risk_prob*100:.0f}%)"

        # 页面标题
        st.markdown(f'<div class="main-title" style="color: {risk_color};">{t["med_advice_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">{t["med_advice_sub"]}</div>', unsafe_allow_html=True)
        
        # 风险状态横幅
        st.markdown(f"""
        <div style="background-color: {risk_bg}; border-left: 8px solid {risk_color}; padding: 15px; border-radius: 8px; margin-bottom: 25px; color: #2C3E50;">
            <span style="font-size: 18px; font-weight: bold; color: {risk_color};">{status_banner_text}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button(t["btn_back_report_med"], key="btn_back_top_med", type="secondary"):
            st.session_state.medical_subpage = "main"
            st.rerun()
            
        st.write("")
        
        # ==========================================
        # 双维度动态医嘱生成矩阵 (Bilingual Matrix)
        # ==========================================
        
        # ------------------------------------------
        # 1. 未成年人分轨 (Age < 18)
        # ------------------------------------------
        if age_val < 18:
            if risk_level == "High":
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #C0392B; background-color: #FDEDEC; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #C0392B; margin-top: 0; font-size: 20px;">Urgent Medical Advice for Guardians/Parents (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>Psychiatric Referral:</b> The subject exhibits severe anxiety and potential Gaming Disorder (GD) tendencies. It is strongly recommended that parents take them to a psychiatric or clinical psychology department for a structured clinical interview (e.g., DSM-5 diagnostic assessment).</li>
                            <li><b>Crisis Limitation:</b> Immediately initiate a "digital detox" plan. Under the guidance of a professional psychologist, temporarily force the reduction of daily recreational screen time to under 1 hour, and strictly prohibit access to any electronic devices after 22:00.</li>
                            <li><b>Empathetic Communication & De-labeling:</b> Strictly avoid aggressive methods such as sudden internet disconnection, physical conflict, or verbal humiliation. Parents need to understand that gaming is a "compensatory mechanism" for the child to escape real-world stress, and trust should be rebuilt through family psychotherapy.</li>
                        </ul>
                        
                        <h3 style="color: #C0392B; font-size: 20px;">Physiological & Behavioral Intervention for the Youth (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>Strict Adherence to the 20-20-20 Rule (Ocular Health):</b> For every 20 minutes of continuous screen time, the youth must look at an object at least 20 feet (about 6 meters) away for at least 20 seconds to prevent acute eye strain and worsening myopia.</li>
                            <li><b>Somatic Symptom Monitoring:</b> Closely monitor for cervical spine compression, tendonitis (gamer's hand), and tension headaches caused by prolonged sitting. A 5-minute break is mandatory for every 30 minutes of play.</li>
                            <li><b>Cognitive Restructuring:</b> Cooperate with the psychologist to identify "compulsive achievement motivation" in games, and try to gradually shift the sense of accomplishment from the virtual world to real-world academics or offline interests.</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #C0392B; background-color: #FDEDEC; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #C0392B; margin-top: 0; font-size: 20px;">给监护人/家长的紧急医学建议 (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>专科转诊评估 (Psychiatric Referral):</b> 受试者已表现出重度焦虑与潜在的游戏障碍 (GD) 倾向。强烈建议家长带其前往精神卫生科或临床心理科，进行结构化临床访谈（如 DSM-5 诊断评估）。</li>
                            <li><b>危机性行为限制 (Crisis Limitation):</b> 立即启动“数字排毒”计划。在专业心理医生指导下，暂时将每日娱乐性屏幕时间强制缩减至 <b>1</b> 小时以内，深夜 <b>22:00</b> 后严格禁止接触任何电子设备。</li>
                            <li><b>共情沟通与去标签化 (De-labeling):</b> 严禁采取粗暴断网、肢体冲突或言语羞辱等方式。家长需理解游戏是孩子逃避现实压力的“代偿手段”，应通过家庭心理治疗重建信任。</li>
                        </ul>
                        
                        <h3 style="color: #C0392B; font-size: 20px;">给青少年本人的生理与行为干预 (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>严格执行 20-20-20 护眼法则 (Ocular Health):</b> 连续注视屏幕每满 <b>20</b> 分钟，必须抬头眺望至少 <b>20</b> 英尺（约 <b>6</b> 米）外的远方目标，持续观望至少 <b>20</b> 秒，以预防急性视疲劳与近视恶化。</li>
                            <li><b>躯体化症状监测 (Somatic Monitoring):</b> 密切关注是否存在因长期久坐导致的颈椎压迫、腱鞘炎（游戏手）及紧张性头痛。每游玩 <b>30</b> 分钟必须强制起立活动 <b>5</b> 分钟。</li>
                            <li><b>认知重构 (Cognitive Restructuring):</b> 配合心理医生，识别自己在游戏中的“强迫性成就动机”，尝试将虚拟世界中的成就感逐步向现实学业或线下兴趣转移。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)
                
            elif risk_level == "Moderate":
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #F1C40F; background-color: #FEF9E7; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #D4AC0D; margin-top: 0; font-size: 20px;">Behavioral Guidance for Guardians/Parents (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>Establish a Screen Time Contract:</b> Parents and youth should negotiate a reasonable screen time limit. Recommended: max 1 hour on weekdays, max 2 hours on weekends.</li>
                            <li><b>Digital-Free Bedroom:</b> Advocate for a "no screens in the bedroom" rule. Avoid placing gaming equipment in the bedroom to reduce the risk of late-night gaming and sleep deprivation at the source.</li>
                            <li><b>High-Quality Alternative Activities:</b> Arrange at least 1 high-quality offline family activity per week (e.g., outdoor sports, board games) to replace virtual rewards with healthy real-world experiences.</li>
                        </ul>
                        
                        <h3 style="color: #D4AC0D; font-size: 20px;">Physical Health Advice for the Youth (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>Ergonomics & Posture:</b> Maintain eye level parallel to or slightly looking down at the screen (15 to 20 degrees) while gaming. Keep feet flat on the floor with knees bent at 90 degrees to protect the rapidly developing spine.</li>
                            <li><b>Active 20-20-20 Rule:</b> Set an alarm to look 20 feet away for 20 seconds every 20 minutes of screen time to relax the ciliary muscles regularly.</li>
                            <li><b>Micro-break Stretching:</b> For every 45 minutes of play, stand up and stretch (chest expansion, neck stretch) for 5 minutes to prevent static muscle injury.</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #F1C40F; background-color: #FEF9E7; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #D4AC0D; margin-top: 0; font-size: 20px;">给监护人/家长的行为引导建议 (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>制定屏幕时间契约 (Behavioral Contract):</b> 家长与青少年共同协商制定合理的屏幕时间。建议工作日每日不超过 <b>1</b> 小时，周末每日不超过 <b>2</b> 小时。</li>
                            <li><b>建立卧室物理边界 (Digital-Free Bedroom):</b> 倡导“卧室无屏幕”原则，避免将游戏设备置于卧室内，从源头上减少深夜偷玩和睡眠剥夺的风险。</li>
                            <li><b>高质量替代性活动 (Alternative Activities):</b> 每周安排至少 <b>1</b> 次高质量的线下亲子活动（如户外运动、桌游），以健康的现实体验替代虚拟奖赏。</li>
                        </ul>
                        
                        <h3 style="color: #D4AC0D; font-size: 20px;">给青少年本人的生理健康建议 (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>视觉与姿势防护 (Ergonomics):</b> 游戏时保持视线平行或轻微向下俯视屏幕 <b>15</b> 度至 <b>20</b> 度，双脚平放于地面，膝关节呈 <b>90</b> 度弯曲，保护处于快速发育期的脊椎。</li>
                            <li><b>主动执行 20-20-20 法则:</b> 设定闹钟，每看屏幕 <b>20</b> 分钟，远眺 <b>20</b> 英尺外 <b>20</b> 秒，让睫状肌得到规律性放松。</li>
                            <li><b>微间歇拉伸:</b> 每游玩 <b>45</b> 分钟，起立进行 <b>5</b> 分钟的扩胸运动和颈部拉伸，预防肌肉静力性损伤。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)
                
            else:
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #2ECC71; background-color: #EAF2F8; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #27AE60; margin-top: 0; font-size: 20px;">Routine Prevention Advice for Guardians/Parents (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>Positive Reinforcement & Trust Building:</b> The subject currently demonstrates excellent self-control. Parents should acknowledge this, maintain the existing trust mechanism, and avoid groundless suspicion or over-monitoring.</li>
                            <li><b>Encourage Diverse Interests:</b> Encourage the child to maintain healthy gaming habits while continuing to explore diverse offline hobbies such as music, art, or science.</li>
                        </ul>
                        
                        <h3 style="color: #27AE60; font-size: 20px;">Routine Health Advice for the Youth (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>Routine Eye Care:</b> Maintain good eye habits, adhere to the 20-20-20 rule, and avoid playing on phones or computers in dimly lit environments.</li>
                            <li><b>Maintain Regular Sleep Patterns:</b> Avoid disrupting daily routines due to holidays or weekends, and ensure sufficient deep sleep.</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #2ECC71; background-color: #EAF2F8; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #27AE60; margin-top: 0; font-size: 20px;">给监护人/家长的常规预防建议 (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 25px;">
                            <li><b>积极强化与信任构建 (Positive Reinforcement):</b> 受试者目前表现出极佳的自我控制力。家长应予以肯定，维持现有的信任机制，避免无端的无端猜忌和过度监控。</li>
                            <li><b>引导探索多元兴趣 (Broadening Horizons):</b> 鼓励孩子在保持健康游戏习惯的同时，继续探索音乐、美术、科学等多元化线下爱好。</li>
                        </ul>
                        
                        <h3 style="color: #27AE60; font-size: 20px;">给青少年本人的常规保健建议 (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li><b>常规用眼卫生 (Routine Eye Care):</b> 维持良好的用眼习惯，坚持 <b>20-20-20</b> 护眼法则，避免在光线昏暗的环境下玩手机或电脑。</li>
                            <li><b>保持规律作息:</b> 避免因假期或周末过度沉迷而打乱日常作息，保持充足的深度睡眠。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)

        # ------------------------------------------
        # 2. 成年人分轨 (Age >= 18)
        # ------------------------------------------
        else:
            if risk_level == "High":
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #C0392B; background-color: #FDEDEC; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #C0392B; margin-top: 0; font-size: 20px;">Clinical Medical Intervention Plan for Adult Patients (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>Clinical Evaluation:</b> Strongly recommend conducting a Hamilton Anxiety Rating Scale (HAMA) assessment and screening for severe depression or addiction comorbidities. If necessary, consider referral to a clinical psychology department for CBT-IA (Cognitive Behavioral Therapy for Internet Addiction).</li>
                            <li style="margin-bottom: 12px;"><b>Dry Eye & Corneal Pathology Management:</b> 
                                <ul>
                                    <li>Prolonged high-intensity screen exposure has severely damaged tear film stability. The 20-20-20 rule must be strictly followed.</li>
                                    <li>It is recommended to regularly use preservative-free artificial tears under the guidance of an ophthalmologist. Avoid staring at screens in dark rooms for long periods.</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>Severe Musculoskeletal Physical Therapy:</b> 
                                <ul>
                                    <li>The subject is highly prone to carpal tunnel syndrome (mouse hand) and loss of cervical lordosis. Immediate physical therapy or rehabilitation evaluation is recommended.</li>
                                    <li>Limit single-session gaming to under 1 hour, followed by 5-10 minutes of resistance stretching.</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>Melatonin Restoration & Sleep Rhythm Reconstruction:</b> Strictly prohibit exposure to blue-light screens within 1 hour before bedtime. If accompanied by severe insomnia, physical therapy or medication is recommended.</li>
                            <li><b>Stress Coping Mechanism Reconstruction:</b> Identify whether gaming is used as the sole "escape mechanism" for occupational stress. Learn cognitive offloading through clinical psychological techniques such as mindfulness meditation or progressive muscle relaxation (PMR).</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #C0392B; background-color: #FDEDEC; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #C0392B; margin-top: 0; font-size: 20px;">给成年患者的临床医学干预方案 (High Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>精神专科评估 (Clinical Evaluation):</b> 强烈建议进行 Hamilton 焦虑量表 (HAMA) 评估，并筛查是否存在重度抑郁或成瘾共病。必要时考虑转诊至临床心理科行 CBT-IA 治疗。</li>
                            <li style="margin-bottom: 12px;"><b>干眼症与角膜病变临床防护 (Dry Eye Management):</b> 
                                <ul>
                                    <li>长期高强度屏幕暴露已导致泪膜稳定性严重受损。必须严格执行 <b>20-20-20 护眼法则</b>。</li>
                                    <li>建议在眼科医师指导下，每日规律使用不含防腐剂的<b>人工泪液</b>（如玻璃酸钠滴眼液），严禁在暗室中长时间注视屏幕。</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>重度骨骼肌肉系统物理治疗 (Ergonomic Intervention):</b> 
                                <ul>
                                    <li>受试者极易出现腕管综合征（鼠标手）及颈椎生理曲度变直。建议立即进行物理治疗或康复科评估。</li>
                                    <li>单次游戏时长强制限制在 <b>1</b> 小时以内，必须起立进行 <b>5-10</b> 分钟的抗阻拉伸。</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>褪黑素重塑与睡眠节律重建 (Sleep Restoration):</b> 睡前 <b>1</b> 小时内严格禁止接触任何蓝光屏幕。若伴有严重失眠，建议结合物理疗法（如经颅磁刺激）或药物调理。</li>
                            <li><b>应激代偿机制重构 (Stress Coping):</b> 识别游戏是否作为职业压力的唯一“逃避机制”，学习通过正念冥想、渐进式肌肉放松（PMR）等临床心理技术进行认知卸载。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)
                
            elif risk_level == "Moderate":
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #F1C40F; background-color: #FEF9E7; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #D4AC0D; margin-top: 0; font-size: 20px;">Health Guidance for Adult Patients (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>Dry Eye Prevention & Eye Strain Control:</b> 
                                <ul>
                                    <li>Execute the <b>20-20-20 rule</b>: look at an object 20 feet away for 20 seconds every 20 minutes of screen time.</li>
                                    <li>Consciously increase blink frequency during gaming to ensure uniform tear film distribution and prevent corneal dryness.</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>Ergonomics & Muscle Protection:</b> 
                                <ul>
                                    <li>Adjust chair height so elbows are at a 90-degree angle with the desk and the top of the monitor is at eye level to reduce static neck muscle contraction.</li>
                                    <li>Avoid continuous gaming for over 2 hours. Stretch for 5 minutes every hour, focusing on wrists and lumbar spine.</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>Sleep Hygiene:</b> Prohibit screen contact 45 minutes before bedtime to block shortwave blue light from inhibiting melatonin secretion. Avoid high-intensity competitive gaming late at night.</li>
                            <li><b>Cognitive Offloading:</b> Perform 5-10 minutes of deep breathing after gaming to help the brain transition smoothly from high excitement to parasympathetic relaxation.</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #F1C40F; background-color: #FEF9E7; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #D4AC0D; margin-top: 0; font-size: 20px;">给成年患者的健康指导建议 (Moderate Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>干眼症预防与视疲劳控制 (Dry Eye Prevention):</b> 
                                <ul>
                                    <li>执行 <b>20-20-20 护眼法则</b>：每注视屏幕 <b>20</b> 分钟，远眺 <b>20</b> 英尺（约 <b>6</b> 米）外 <b>20</b> 秒。</li>
                                    <li>游戏时有意识地提高眨眼频率，确保泪膜均匀分布，防止角膜干燥。</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>人体工学与肌肉防护 (Ergonomics):</b> 
                                <ul>
                                    <li>调整座椅高度，使手肘与桌面呈 <b>90</b> 度直角，显示器上沿与视线等高，减少颈部肌肉持续性静力收缩。</li>
                                    <li>避免单次连续游戏超过 <b>2</b> 小时。每隔 <b>1</b> 小时进行一次 <b>5</b> 分钟的全身拉伸，重点放松腕部和腰椎。</li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 12px;"><b>睡眠节律维护 (Sleep Hygiene):</b> 睡前 <b>45</b> 分钟内禁止接触电子屏幕，阻断短波蓝光对松果体分泌褪黑素的抑制，避免在深夜进行高强度竞技类游戏。</li>
                            <li><b>积极压力宣泄 (Cognitive Offloading):</b> 游戏结束后，进行 <b>5-10</b> 分钟的深呼吸，帮助大脑从游戏的高兴奋状态平稳过渡到副交感神经主导的放松状态。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)
                
            else:
                if lang == "en":
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #2ECC71; background-color: #EAF2F8; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #27AE60; margin-top: 0; font-size: 20px;">Routine Health Advice for Adult Subjects (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>Maintain Healthy Boundaries:</b> The subject currently maintains a good balance between life and entertainment. It is recommended to keep the current weekly gaming hours as a healthy way to decompress.</li>
                            <li style="margin-bottom: 12px;"><b>Routine Eye & Posture Care:</b> 
                                <ul>
                                    <li>Occasionally apply the <b>20-20-20 rule</b> during long working or gaming sessions.</li>
                                    <li>Maintain good ergonomic sitting posture, avoiding prolonged slouching.</li>
                                </ul>
                            </li>
                            <li><b>Routine Follow-up:</b> No special clinical intervention is required. Routine self-assessment of mental and physical health is recommended every six months.</li>
                        </ul>
                    </div>
                    """
                else:
                    intervention_html = f"""
                    <div class="card-intervention" style="border-left: 8px solid #2ECC71; background-color: #EAF2F8; padding: 30px; border-radius: 10px; color: #2C3E50; margin-bottom: 20px;">
                        <h3 style="color: #27AE60; margin-top: 0; font-size: 20px;">给成年受试者的常规保健建议 (Low Risk)</h3>
                        <ul style="line-height: 1.8; font-size: 15px; margin-bottom: 0;">
                            <li style="margin-bottom: 12px;"><b>维持健康的娱乐边界 (Healthy Boundaries):</b> 受试者目前生活与娱乐平衡良好。建议继续保持当前的每周游戏时长，将其作为健康的减压手段。</li>
                            <li style="margin-bottom: 12px;"><b>常规用眼与姿势保健:</b> 
                                <ul>
                                    <li>在长时间办公或游戏时，偶尔运用 <b>20-20-20</b> 护眼法则放松眼部。</li>
                                    <li>保持良好的人体工学坐姿，避免长时间低头或瘫坐。</li>
                                </ul>
                            </li>
                            <li><b>常规随访:</b> 无需特殊临床干预，建议每半年进行一次常规的心理与生理健康自测。</li>
                        </ul>
                    </div>
                    """
                st.markdown(intervention_html, unsafe_allow_html=True)
        
        st.write("---")
        if st.button(t["btn_back_report_med"], key="btn_back_bottom_med", type="secondary", use_container_width=True):
            st.session_state.medical_subpage = "main"
            st.rerun()



    # =========================================================================
    # 🖥️ PLAYER & MEDICAL - MAIN DASHBOARD (主评估页面)
    # =========================================================================
    else:
        # 1:2 Layout Configuration
        col_left, col_right = st.columns([1, 2], gap="large")

        # ==========================================
        # 👈 Left Column: Feature Input (Ratio: 1)
        # ==========================================
        with col_left:
            st.markdown(f'<div class="section-header">{t["panel_title"]}</div>', unsafe_allow_html=True)
            
            with st.container():
                # 1. Age Group Selection
                age_group_opts = [
                    '10-17 (Adolescent / School & Growth Stage)', 
                    '18-30 (Young Adult / College & Early Career Stage)', 
                    '31-50 (Adult / Established Family & Career Stage)', 
                    '51-80 (Senior / Retirement & Leisure Stage)'
                ] if lang == "en" else [
                    '10-17岁 (青少年 / 学业与成长阶段)', 
                    '18-30岁 (青年 / 高校与职业早期阶段)', 
                    '31-50岁 (成年 / 职业骨干与家庭阶段)', 
                    '51-80岁 (长者 / 退休与闲暇阶段)'
                ]
                
                age_group = st.selectbox(t["q_age_group"], options=age_group_opts, index=1)
                
                # Dynamic Age Bounds Mapping
                age_bounds = {
                    age_group_opts[0]: {'min': 10, 'max': 17, 'default': 15},
                    age_group_opts[1]: {'min': 18, 'max': 30, 'default': 22},
                    age_group_opts[2]: {'min': 31, 'max': 50, 'default': 35},
                    age_group_opts[3]: {'min': 51, 'max': 80, 'default': 60}
                }
                current_bounds = age_bounds[age_group]
                
                # Exact Age Slider
                age_selected = st.slider(
                    t["q_age_exact"],
                    min_value=current_bounds['min'],
                    max_value=current_bounds['max'],
                    value=current_bounds['default'],
                    step=1
                )
                
                # Gender Selection
                gender_selected = st.radio(t["q_gender"], options=["Male", "Female"] if lang == "en" else ["男 (Male)", "女 (Female)"], index=0, horizontal=True)
                gender_code = "Female" if "Female" in gender_selected or "女" in gender_selected else "Male"
                
                # Weekly Gaming Hours Slider
                hours_selected = st.slider(t["q_hours"], min_value=1, max_value=80, value=25, step=1)
                
                # Game Type Selection
                game_type_opts = ["Single-player Games", "Multiplayer / Online Games"] if lang == "en" else ["单机游戏 (Single-player)", "联机/网络游戏 (Multiplayer)"]
                game_type_selected = st.selectbox(t["q_game_type"], options=game_type_opts, index=0)
                game_type_code = "Single-player Games" if "Single" in game_type_selected or "单机" in game_type_selected else "Multiplayer / Online Games"
                
                # 🌐 5. Actual Social Behavior
                st.write("")
                st.markdown(f"**{t['q_behavior']}**")
                
                # Behavior Option 1: Solo Play
                is_solo_active = "solo" in st.session_state.selected_behaviors
                solo_label = f"🔴 {t['behavior_solo']}" if is_solo_active else t['behavior_solo']
                if st.button(solo_label, key="btn_beh_solo", use_container_width=True):
                    if "solo" in st.session_state.selected_behaviors:
                        st.session_state.selected_behaviors.remove("solo")
                    else:
                        st.session_state.selected_behaviors.append("solo")
                    st.rerun()
                    
                # Behavior Option 2: Active Socializing
                is_social_active = "social" in st.session_state.selected_behaviors
                social_label = f"🔴 {t['behavior_social']}" if is_social_active else t['behavior_social']
                if st.button(social_label, key="btn_beh_social", use_container_width=True):
                    if "social" in st.session_state.selected_behaviors:
                        st.session_state.selected_behaviors.remove("social")
                    else:
                        st.session_state.selected_behaviors.append("social")
                    st.rerun()
                
                st.write("")
                submit_btn = st.button(t["btn_eval"], type="primary", use_container_width=True)

        # ==========================================
        # 👉 Right Column: Results & Diagnosis (Ratio: 2)
        # ==========================================
        with col_right:
            st.markdown(f'<div class="section-header">{t["report_title"]}</div>', unsafe_allow_html=True)
            
            if submit_btn or ("last_risk_prob" in st.session_state):
                # Save calculation results in session state to prevent loss when toggling subpages
                if submit_btn:
                    with st.spinner(t["evaluating"]):
                        time.sleep(0.6)
                        
                        # Dynamic Logic Engine
                        base_value = 35.0
                        drivers = []
                        protectors = [t["prot_self"]]
                        
                        # Age Logic
                        if age_selected <= 17:
                            drivers.append(f"<b>{t['q_age_group']} ({age_selected} y/o)</b>: " + ("青少年学业压力重，情绪调节机制尚在发育中。" if lang == "zh" else "Adolescents face school stress and ongoing emotional brain development."))
                        elif age_selected <= 30:
                            drivers.append(f"<b>{t['q_age_group']} ({age_selected} y/o)</b>: " + ("青年面临职业转型与社会角色适应期。" if lang == "zh" else "Young adults face career transitions and social adjustments."))
                        else:
                            protectors.append(f"<b>{t['q_age_group']} ({age_selected} y/o)</b>: " + ("成年人通常具备更成熟的情绪缓冲机制。" if lang == "zh" else "Mature adults possess established coping mechanisms."))
                        
                        # Gender Logic
                        if gender_code == "Female":
                            drivers.append("<b>" + ("性别敏感度" if lang == "zh" else "Gender Sensitivity") + "</b>: " + ("统计学上女性对情绪压力易感性略高。" if lang == "zh" else "Statistically higher sensitivity to internal stressors."))
                        else:
                            protectors.append("<b>" + ("性别因素" if lang == "zh" else "Gender Factor") + "</b>: " + ("统计学上与较低的焦虑报告率相关。" if lang == "zh" else "Associated with lower baseline reported anxiety."))
                        
                        # Hours Logic
                        if hours_selected > 40:
                            drivers.append(f"<b>" + ("超长游戏时间" if lang == "zh" else "Excessive Screen Time") + f" ({hours_selected}h/wk)</b>: " + ("严重剥夺深度睡眠与线下现实社交。" if lang == "zh" else "Severely deprives sleep and offline socialization."))
                        elif hours_selected > 20:
                            drivers.append(f"<b>" + ("中度游戏时间" if lang == "zh" else "Moderate-High Screen Time") + f" ({hours_selected}h/wk)</b>: " + ("游戏时间偏高，需防范慢性身体疲劳。" if lang == "zh" else "Slightly elevated; monitor for physical fatigue."))
                        else:
                            protectors.append(f"<b>" + ("健康游戏限制" if lang == "zh" else "Healthy Gaming Limits") + f" ({hours_selected}h/wk)</b>: " + ("时间控制优秀，属于健康的娱乐消遣。" if lang == "zh" else "Excellent time management; kept as a healthy hobby."))
                        
                        # Social Behavior Logic
                        has_solo = "solo" in st.session_state.selected_behaviors
                        has_social = "social" in st.session_state.selected_behaviors
                        
                        if has_solo and has_social:
                            protectors.append("<b>" + ("混合健康游玩" if lang == "zh" else "Healthy Hybrid Playstyle") + "</b>: " + ("既享受独处沉浸，又乐于社区分享，形成极佳的情绪缓冲。" if lang == "zh" else "Enjoys focused solo play while actively sharing online. Excellent buffer!"))
                        elif has_solo:
                            if game_type_code == "Multiplayer / Online Games":
                                drivers.append("<b>" + ("网游孤狼状态" if lang == "zh" else "Online Game but Solo Play") + "</b>: " + ("在多人网游中缺乏社交支持，独自面对竞技压力。" if lang == "zh" else "Facing competitive online matchmaking without friends to buffer stress."))
                            else:
                                drivers.append("<b>" + ("单机孤狼状态" if lang == "zh" else "Pure Solo Gaming") + "</b>: " + ("完全孤立游玩，缺乏外部社区交流和情绪宣泄渠道。" if lang == "zh" else "Immersing in single-player worlds completely isolated limits emotional venting."))
                        elif has_social:
                            if game_type_code == "Single-player Games":
                                protectors.append("<b>" + ("单机社区共鸣" if lang == "zh" else "Single-player with Community Connection") + "</b>: " + ("在社交媒体分享单机心得，连接了温暖的同好社区。" if lang == "zh" else "Sharing achievements online connects you to a warm community."))
                            else:
                                protectors.append("<b>" + ("高社交游戏" if lang == "zh" else "Highly Social Gaming") + "</b>: " + ("将游戏转化为虚拟茶话会，有效稀释日常焦虑。" if lang == "zh" else "Turns gaming into a virtual lounge, shielding against anxiety."))
                        
                        # Risk calculation
                        prob = base_value
                        if age_selected <= 17: prob += 15
                        elif age_selected <= 30: prob += 10
                        else: prob -= 10
                        if gender_code == "Female": prob += 5
                        if hours_selected > 40: prob += 30
                        elif hours_selected > 20: prob += 10
                        else: prob -= 15
                        if has_solo and has_social: prob -= 2
                        elif has_solo: prob += 8
                        elif has_social: prob -= 5
                        risk_prob = np.clip(prob, 5.0, 95.0) / 100.0
                        
                        # Store in session state
                        st.session_state.last_risk_prob = risk_prob
                        st.session_state.last_drivers = drivers
                        st.session_state.last_protectors = protectors
                        st.session_state.last_hours = hours_selected
                        st.session_state.last_age = age_selected
                        st.session_state.last_gender = gender_code
                        st.session_state.last_has_solo = has_solo
                        st.session_state.last_has_social = has_social
                
                # Retrieve from session state
                risk_prob = st.session_state.last_risk_prob
                drivers = st.session_state.last_drivers
                protectors = st.session_state.last_protectors
                hours_selected = st.session_state.last_hours
                age_selected = st.session_state.last_age
                gender_code = st.session_state.last_gender
                has_solo = st.session_state.last_has_solo
                has_social = st.session_state.last_has_social

                # ------------------------------------------
                # Feature 1: Risk Screening Results
                # ------------------------------------------
                if identity == "Player":
                    if risk_prob >= 0.70:
                        st.markdown(f'<div class="card-risk-high"><h3 style="color: #C0392B; margin-top:0; font-size: 24px;">{t["risk_high_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["risk_high_desc"]}</p></div>', unsafe_allow_html=True)
                    elif risk_prob >= 0.40:
                        st.markdown(f'<div class="card-risk-med"><h3 style="color: #D4AC0D; margin-top:0; font-size: 24px;">{t["risk_med_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["risk_med_desc"]}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="card-risk-low"><h3 style="color: #27AE60; margin-top:0; font-size: 24px;">{t["risk_low_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["risk_low_desc"]}</p></div>', unsafe_allow_html=True)
                else:
                    if risk_prob >= 0.70:
                        st.markdown(f'<div class="card-risk-high"><h3 style="color: #C0392B; margin-top:0; font-size: 24px;">{t["cli_high_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["cli_high_desc"]}</p></div>', unsafe_allow_html=True)
                    elif risk_prob >= 0.40:
                        st.markdown(f'<div class="card-risk-med"><h3 style="color: #D4AC0D; margin-top:0; font-size: 24px;">{t["cli_med_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["cli_med_desc"]}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="card-risk-low"><h3 style="color: #27AE60; margin-top:0; font-size: 24px;">{t["cli_low_title"]} ({risk_prob*100:.0f}%)</h3><p style="font-size: 15px; line-height: 1.6;">{t["cli_low_desc"]}</p></div>', unsafe_allow_html=True)

                # ------------------------------------------
                # Feature 2: Individual Diagnosis
                # ------------------------------------------
                if identity == "Player":
                    # ⚖️ Warm Balance Scale for Players
                    st.markdown(f"<h3 style='color: #D35400; margin-top:15px; font-size: 22px;'>{t['scale_title']}</h3>", unsafe_allow_html=True)
                    st.write(t["scale_sub"])
                    st.write("")
                    
                    scale_col_left, scale_col_right = st.columns(2)
                    
                    with scale_col_left:
                        left_items_html = "".join([f"<p style='margin-bottom: 12px; line-height: 1.5; font-size: 14px;'><span class='icon-warning'>⚠️</span> {d}</p>" for d in drivers])
                        st.markdown(f'<div class="scale-box-left"><h4 style="color: #C0392B; margin-top:0; text-align:center; border-bottom: 1px solid #F5B7B1; padding-bottom: 8px; margin-bottom: 15px;">{t["plate_left"]}</h4>{left_items_html}</div>', unsafe_allow_html=True)
                        
                    with scale_col_right:
                        right_items_html = "".join([f"<p style='margin-bottom: 12px; line-height: 1.5; font-size: 14px;'><span class='icon-success'>✅</span> {p}</p>" for p in protectors])
                        st.markdown(f'<div class="scale-box-right"><h4 style="color: #2980B9; margin-top:0; text-align:center; border-bottom: 1px solid #AED6F1; padding-bottom: 8px; margin-bottom: 15px;">{t["plate_right"]}</h4>{right_items_html}</div>', unsafe_allow_html=True)
                        
                    st.write("")
                    if len(drivers) > len(protectors):
                        st.markdown(f'<div class="scale-status" style="background-color: #FDEDEC; color: #C0392B; border: 1px solid #FADBD8;">{t["scale_left_tilt"]}</div>', unsafe_allow_html=True)
                    elif len(drivers) < len(protectors):
                        st.markdown(f'<div class="scale-status" style="background-color: #EAF2F8; color: #2980B9; border: 1px solid #D4E6F1;">{t["scale_right_tilt"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="scale-status" style="background-color: #FEF9E7; color: #D4AC0D; border: 1px solid #FCF3CF;">{t["scale_balance"]}</div>', unsafe_allow_html=True)
                    
                    # 🌟 玩家端：绿色虚线过渡框 + 跳转按钮
                    st.write("")
                    st.markdown(f"""
                    <div style="background-color: #E8F8F5; border: 2px dashed #2ECC71; padding: 20px; border-radius: 12px; text-align: center; margin-top: 15px;">
                        <p style="margin: 0; font-size: 15px; color: #27AE60; font-weight: bold;">
                            {t["heal_box_prompt"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    if st.button(t["btn_next_page"], key="btn_go_to_healing", use_container_width=True, type="primary"):
                        st.session_state.player_subpage = "recommendations"
                        st.rerun()
                
                else:
                    # 📊 Professional Dynamic Matplotlib Radar Chart for Medical Staff
                    st.markdown(f"<h3 style='color: #1F618D; margin-top:15px; font-size: 22px;'>{t['cli_chart_title']}</h3>", unsafe_allow_html=True)
                    st.write(t["cli_chart_sub"])
                    
                    # --- Dynamic Clinical Feature Mapping Engine ---
                    bio_vul = 20
                    if age_selected <= 17: bio_vul += 30
                    elif age_selected <= 30: bio_vul += 15
                    if gender_code == "Female": bio_vul += 15
                    
                    cog_load = 15
                    if hours_selected > 40: cog_load += 65
                    elif hours_selected > 20: cog_load += 35
                    else: cog_load += 10
                    
                    soc_def = 20
                    if has_solo and not has_social: soc_def += 60
                    elif has_solo and has_social: soc_def += 15
                    elif not has_solo and has_social: soc_def += 5
                    
                    sleep_imp = 15
                    if hours_selected > 40: sleep_imp += 65
                    elif hours_selected > 20: sleep_imp += 35
                    else: sleep_imp += 10
                    if age_selected <= 17: sleep_imp += 10
                    sleep_imp = min(sleep_imp, 95)
                    
                    escapism = 20
                    if hours_selected > 40 and has_solo: escapism += 65
                    elif hours_selected > 20 and has_solo: escapism += 45
                    elif hours_selected > 40 and has_social: escapism += 35
                    else: escapism += 15
                    escapism = min(escapism, 95)
                    
                    # --- Plotting the Radar Chart dynamically ---
                    categories = [
                        'Bio-Vulnerability\n', 
                        'Cognitive Load\n', 
                        'Social Deficit\n', 
                        'Sleep Impairment\n', 
                        'Escapism Dependency\n'
                    ]
                    N = len(categories)
                    
                    patient_values = [bio_vul, cog_load, soc_def, sleep_imp, escapism]
                    baseline_values = [25, 30, 30, 25, 35]
                    
                    patient_plot = patient_values + [patient_values[0]]
                    baseline_plot = baseline_values + [baseline_values[0]]
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]
                    
                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
                    
                    plt.xticks(angles[:-1], categories, color='#2C3E50', size=9, fontweight='bold')
                    ax.set_rlabel_position(0)
                    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#7F8C8D", size=8)
                    plt.ylim(0, 100)
                    ax.grid(color='#BDC3C7', linestyle='--', linewidth=0.6)
                    
                    # Plot Baseline (Green)
                    ax.plot(angles, baseline_plot, linewidth=1.5, linestyle='solid', color='#2ECC71', label='Healthy Control Baseline')
                    ax.fill(angles, baseline_plot, '#2ECC71', alpha=0.1)
                    
                    # Plot Patient State (Orange)
                    ax.plot(angles, patient_plot, linewidth=2, linestyle='solid', color='#E67E22', label='Patient Current State')
                    ax.fill(angles, patient_plot, '#E74C3C', alpha=0.2)
                    
                    ax.spines['polar'].set_visible(False)
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=8)
                    
                    st.pyplot(fig)
                    
                    # --- 动态双语临床诊断参考 (Clinical Insights) ---
                    if lang == "en":
                        insights_html = f"""
                        <div style="background-color: #F8F9F9; padding: 20px; border-radius: 8px; border: 1px solid #E5E7E9; margin-top: 15px;">
                            <h4 style="margin: 0 0 12px 0; font-size: 18px; color: #1F618D; font-weight: bold;">
                                Clinical Insights
                            </h4>
                            <p style="margin:0; font-size: 16px; color: #2C3E50; line-height: 1.7;">
                                • <b>Bio-Vulnerability:</b> Score {bio_vul}%. Based on the subject's age and gender physiological vulnerability.<br>
                                • <b>Cognitive Load:</b> Score {cog_load}%. Driven by high-frequency weekly screen exposure and neural fatigue.<br>
                                • <b>Social Deficit:</b> Score {soc_def}%. Reflects offline social isolation and solo gaming habits.<br>
                                • <b>Sleep Impairment:</b> Score {sleep_imp}%. Potential inhibition of melatonin secretion due to high-frequency nighttime screen exposure.<br>
                                • <b>Escapism Dependency:</b> Score {escapism}%. Evaluates the subject's tendency for mental compensation, using gaming as the sole stress outlet.
                            </p>
                        </div>
                        """
                    else:
                        insights_html = f"""
                        <div style="background-color: #F8F9F9; padding: 20px; border-radius: 8px; border: 1px solid #E5E7E9; margin-top: 15px;">
                            <h4 style="margin: 0 0 12px 0; font-size: 18px; color: #1F618D; font-weight: bold;">
                                临床诊断参考 (Clinical Insights)
                            </h4>
                            <p style="margin:0; font-size: 16px; color: #2C3E50; line-height: 1.7;">
                                • <b>生理易感性 (Bio-Vulnerability):</b> 评分 {bio_vul}%。基于受试者年龄与性别生理易感性。<br>
                                • <b>认知行为负荷 (Cognitive Load):</b> 评分 {cog_load}%。受每周高频屏幕暴露与神经疲劳驱动。<br>
                                • <b>社会支持缺失 (Social Deficit):</b> 评分 {soc_def}%。反映线下社交隔离与孤狼游玩习惯。<br>
                                • <b>睡眠受损度 (Sleep Impairment):</b> 评分 {sleep_imp}%。高频晚间屏幕暴露对褪黑素分泌的潜在抑制。<br>
                                • <b>情绪代偿依赖 (Escapism Dependency):</b> 评分 {escapism}%。评估受试者将游戏作为唯一压力宣泄口的精神代偿倾向。
                            </p>
                        </div>
                        """
                    
                    st.markdown(insights_html, unsafe_allow_html=True)




                
                    # 🌟 医生端：蓝色虚线过渡框 + 跳转按钮
                    st.write("")
                    st.markdown(f"""
                    <div style="background-color: #EBF5FB; border: 2px dashed #2980B9; padding: 20px; border-radius: 12px; text-align: center; margin-top: 15px;">
                        <p style="margin: 0; font-size: 15px; color: #2980B9; font-weight: bold;">
                            {t["med_box_prompt"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    if st.button(t["btn_next_page_med"], key="btn_go_to_intervention", use_container_width=True, type="primary"):
                        st.session_state.medical_subpage = "intervention"
                        st.rerun()
                
            else:
                st.info(t["info_prompt"])
