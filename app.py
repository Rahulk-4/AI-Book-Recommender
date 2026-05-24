import pandas as pd
import streamlit as st
import time
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Book Recommender", layout="wide", page_icon="📚")

# ═══════════════════════════════════════════════════════════
#  CSS — all styles including new features
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,.stApp{background:#06060a!important;color:#f0ece4;font-family:'DM Sans',sans-serif;cursor:none!important;}
*,a,button,select,input,textarea,label,[role="button"]{cursor:none!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:#06060a;}
::-webkit-scrollbar-thumb{background:#1e1e26;border-radius:10px;}
::-webkit-scrollbar-thumb:hover{background:#e50914;}
.stApp::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.022) 2px,rgba(0,0,0,.022) 4px);pointer-events:none;z-index:0;}

/* ── ORBS ── */
.orb-layer{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(100px);opacity:.1;animation:orbFloat linear infinite;}
.orb-a{width:600px;height:600px;background:#e50914;top:-220px;left:-160px;animation-duration:22s;}
.orb-b{width:380px;height:380px;background:#ff4500;bottom:-140px;right:-100px;animation-duration:28s;animation-direction:reverse;}
.orb-c{width:260px;height:260px;background:#c0392b;top:44%;left:58%;animation-duration:36s;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}25%{transform:translate(55px,-70px) scale(1.1);}50%{transform:translate(-35px,40px) scale(.94);}75%{transform:translate(65px,55px) scale(1.06);}}

/* ── SECTION TITLE ── */
.sec-title{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:#f0ece4;margin:1.5rem 0 1rem;}
.sec-title em{color:#e50914;font-style:italic;}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #12121a!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:.75rem!important;letter-spacing:2px!important;text-transform:uppercase!important;color:#282832!important;padding:.9rem 1.4rem!important;border-radius:0!important;border-bottom:2px solid transparent!important;background:transparent!important;transition:color .25s ease!important;}
.stTabs [data-baseweb="tab"]:hover{color:#f0ece4!important;}
.stTabs [data-baseweb="tab"][aria-selected=true]{color:#e50914!important;border-bottom:2px solid #e50914!important;}
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}

/* ── BOOK CARDS ── */
.cards-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:10px;}
.bcard{background:#0e0e14;border:1px solid #18181e;border-radius:14px;overflow:hidden;position:relative;opacity:0;animation:cardIn .75s cubic-bezier(.22,1,.36,1) both;transition:transform .32s cubic-bezier(.22,1,.36,1),border-color .32s ease,box-shadow .32s ease;}
.bcard:nth-child(1){animation-delay:.06s;}.bcard:nth-child(2){animation-delay:.14s;}.bcard:nth-child(3){animation-delay:.22s;}.bcard:nth-child(4){animation-delay:.30s;}.bcard:nth-child(5){animation-delay:.38s;}.bcard:nth-child(6){animation-delay:.46s;}
@keyframes cardIn{from{opacity:0;transform:perspective(520px) rotateX(18deg) translateY(28px) scale(.95);}to{opacity:1;transform:perspective(520px) rotateX(0) translateY(0) scale(1);}}
.bcard::before{content:'';position:absolute;inset:0;border-radius:14px;background:linear-gradient(130deg,rgba(229,9,20,.2),transparent 55%);opacity:0;transition:opacity .35s ease;z-index:1;pointer-events:none;}
.bcard:hover::before{opacity:1;}
.bcard:hover{border-color:rgba(229,9,20,.38);transform:translateY(-11px) scale(1.025);box-shadow:0 0 0 1px rgba(229,9,20,.18),0 24px 48px rgba(229,9,20,.15),0 48px 90px rgba(0,0,0,.55);}
.bcover{aspect-ratio:2/3;background:linear-gradient(145deg,#0d0d16,#07070e)!important;display:flex;align-items:center;justify-content:center;font-size:38px;position:relative;overflow:hidden;}
.bcover img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .65s cubic-bezier(.22,1,.36,1);background:#0d0d16!important;}
.bcard:hover .bcover img{transform:scale(1.09);}
.bcover-spine{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:16px;background:linear-gradient(160deg,#111120,#080810);}
.bcover-spine-title{font-family:'Playfair Display',serif;font-size:11px;font-weight:700;color:#f0ece4;text-align:center;line-height:1.4;opacity:.8;}
.bcover-spine-line{width:24px;height:1px;background:linear-gradient(90deg,transparent,#e50914,transparent);}
.bcover-spine-icon{font-size:28px;opacity:.6;}
.bcover-shimmer{position:absolute;inset:0;background:linear-gradient(108deg,transparent 38%,rgba(255,255,255,.04) 50%,transparent 62%);background-size:250% 100%;animation:shimmer 2.8s ease-in-out infinite;}
@keyframes shimmer{0%{background-position:-250% 0;}100%{background-position:250% 0;}}
.bcover-shadow{position:absolute;bottom:0;left:0;right:0;height:60%;background:linear-gradient(to top,#0e0e14 0%,transparent 100%);}
.bcardBody{padding:11px;position:relative;z-index:2;}
.bcard-title{font-family:'Playfair Display',serif;font-size:13px;font-weight:700;color:#f0ece4;line-height:1.3;margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.bcard-author{font-size:10px;color:#2e2e3a;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px;}
.bcard-genre{display:inline-block;font-size:9.5px;font-weight:700;color:#e50914;background:rgba(229,9,20,.09);border:1px solid rgba(229,9,20,.18);border-radius:20px;padding:2px 9px;text-transform:uppercase;letter-spacing:1px;margin-bottom:9px;}
.bcard-rating{display:flex;align-items:center;gap:7px;}
.brating-track{flex:1;height:2px;background:#14141c;border-radius:10px;overflow:hidden;}
.brating-fill{height:100%;border-radius:10px;background:linear-gradient(90deg,#e50914,#ff7055);animation:ratingIn 1.3s cubic-bezier(.22,1,.36,1) both;}
.bcard:nth-child(1) .brating-fill{animation-delay:.55s;}.bcard:nth-child(2) .brating-fill{animation-delay:.63s;}.bcard:nth-child(3) .brating-fill{animation-delay:.71s;}.bcard:nth-child(4) .brating-fill{animation-delay:.79s;}.bcard:nth-child(5) .brating-fill{animation-delay:.87s;}.bcard:nth-child(6) .brating-fill{animation-delay:.95s;}
@keyframes ratingIn{from{width:0;}}
.brating-num{font-size:11px;font-weight:700;color:#e50914;min-width:24px;text-align:right;}

/* ── MOOD BADGES ── */
.mood-row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 18px;}
.mood-badge{font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;padding:6px 16px;border-radius:20px;border:1px solid #1e1e28;background:#0e0e14;color:#444;transition:all .25s ease;}
.mood-badge:hover,.mood-badge.active{border-color:rgba(229,9,20,.5);color:#e50914;background:rgba(229,9,20,.09);}

/* ── BOOK OF THE DAY ── */
.botd{background:linear-gradient(135deg,#0e0e14,#14101a);border:1px solid #1e1e28;border-radius:14px;padding:22px 24px;display:flex;gap:20px;align-items:center;margin:16px 0;position:relative;overflow:hidden;}
.botd::before{content:'BOOK OF THE DAY';position:absolute;top:14px;right:16px;font-size:8px;letter-spacing:3px;color:#e50914;font-weight:700;border:1px solid rgba(229,9,20,.2);border-radius:20px;padding:3px 10px;}
.botd-cover{width:80px;height:120px;border-radius:8px;background:linear-gradient(145deg,#111120,#07070e);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:32px;border:1px solid #1e1e28;overflow:hidden;}
.botd-cover img{width:100%;height:100%;object-fit:cover;border-radius:8px;background:#0d0d16;}
.botd-title{font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:900;color:#f0ece4;margin-bottom:4px;}
.botd-author{font-size:11px;color:#444;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;}
.botd-desc{font-size:12px;color:#555;line-height:1.6;margin-bottom:10px;}
.botd-stars{color:#e50914;font-size:13px;}
.botd-num{font-size:13px;font-weight:700;color:#e50914;}

/* ── QUICK STRIP ── */
.qstrip{display:flex;gap:10px;margin:12px 0 20px;flex-wrap:wrap;}
.qchip{background:#0e0e14;border:1px solid #18181e;border-radius:20px;padding:6px 14px;font-size:10px;letter-spacing:1px;color:#444;display:flex;align-items:center;gap:6px;}
.qchip span{color:#e50914;font-weight:700;}

/* ── BUTTON ── */
.stButton>button{background:#e50914!important;color:white!important;border:none!important;border-radius:8px!important;height:3.1rem!important;font-size:.78rem!important;padding:0 2rem!important;font-weight:700!important;font-family:'DM Sans',sans-serif!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:transform .22s ease,box-shadow .22s ease!important;box-shadow:0 4px 20px rgba(229,9,20,.28)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 10px 30px rgba(229,9,20,.45)!important;}
.stButton>button:active{transform:scale(.97)!important;}

/* ── INPUTS ── */
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea textarea{background:#0e0e14!important;border:1px solid #1e1e28!important;border-radius:8px!important;color:#f0ece4!important;font-family:'DM Sans',sans-serif!important;transition:border-color .25s,box-shadow .25s!important;}
.stSelectbox>div>div:focus-within,.stTextInput>div>div>input:focus,.stTextArea textarea:focus{border-color:rgba(229,9,20,.5)!important;box-shadow:0 0 0 3px rgba(229,9,20,.09)!important;}

/* ── STAT CARDS ── */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px;}
.stat-card{background:#0e0e14;border:1px solid #18181e;border-radius:12px;padding:20px 16px;text-align:center;transition:border-color .3s,transform .3s;animation:fadeUp .6s ease both;}
.stat-card:nth-child(1){animation-delay:.1s;}.stat-card:nth-child(2){animation-delay:.18s;}.stat-card:nth-child(3){animation-delay:.26s;}.stat-card:nth-child(4){animation-delay:.34s;}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
.stat-card:hover{border-color:rgba(229,9,20,.25);transform:translateY(-4px);}
.stat-num{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:900;color:#e50914;display:block;line-height:1;}
.stat-lbl{font-size:9.5px;color:#282832;text-transform:uppercase;letter-spacing:3px;margin-top:6px;display:block;}

/* ── DATAFRAME / ALERTS ── */
.stDataFrame{border:1px solid #18181e!important;border-radius:10px!important;overflow:hidden;}
.stSuccess{background:rgba(229,9,20,.07)!important;border:1px solid rgba(229,9,20,.2)!important;border-radius:8px!important;}
.stChatMessage{background:#0e0e14!important;border:1px solid #18181e!important;border-radius:12px!important;animation:chatSlide .3s ease both!important;}
@keyframes chatSlide{from{opacity:0;transform:translateX(-8px);}to{opacity:1;transform:translateX(0);}}
label,.stSelectbox label,.stTextInput label{color:#333!important;font-size:.75rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}

/* ══════════════════════════════════════════════
   ❤️  FAVOURITES
══════════════════════════════════════════════ */
.fav-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-top:12px;}
.fav-card{background:#0e0e14;border:1px solid #18181e;border-radius:12px;padding:14px;display:flex;gap:12px;align-items:center;animation:fadeUp .5s ease both;transition:border-color .3s,transform .3s;}
.fav-card:hover{border-color:rgba(229,9,20,.3);transform:translateY(-3px);}
.fav-icon{font-size:1.8rem;flex-shrink:0;}
.fav-info{}
.fav-title{font-family:'Playfair Display',serif;font-size:13px;font-weight:700;color:#f0ece4;margin-bottom:3px;}
.fav-meta{font-size:10px;color:#333;text-transform:uppercase;letter-spacing:1px;}
.fav-rating{font-size:10px;color:#e50914;margin-top:3px;font-weight:700;}
.fav-empty{text-align:center;padding:48px 20px;color:#282832;font-size:13px;letter-spacing:1px;}
.fav-empty-icon{font-size:3rem;margin-bottom:12px;display:block;opacity:.3;}

/* ══════════════════════════════════════════════
   📖  BOOK DETAILS POPUP (expander style)
══════════════════════════════════════════════ */
.detail-box{background:linear-gradient(135deg,#0e0e14,#12101c);border:1px solid #1e1e28;border-radius:16px;padding:24px;margin-top:16px;display:flex;gap:24px;animation:fadeUp .5s ease both;}
.detail-cover{width:120px;height:180px;border-radius:10px;flex-shrink:0;overflow:hidden;background:linear-gradient(145deg,#111120,#080810);display:flex;align-items:center;justify-content:center;font-size:3rem;border:1px solid #1e1e28;}
.detail-cover img{width:100%;height:100%;object-fit:cover;background:#0d0d16;}
.detail-info{flex:1;}
.detail-title{font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:900;color:#f0ece4;margin-bottom:4px;}
.detail-author{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;}
.detail-tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;}
.detail-tag{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:3px 11px;border-radius:20px;border:1px solid rgba(229,9,20,.2);color:#e50914;background:rgba(229,9,20,.07);}
.detail-desc{font-size:13px;color:#484855;line-height:1.8;margin-bottom:14px;}
.detail-stars{color:#e50914;font-size:16px;letter-spacing:2px;}
.detail-rating-num{font-size:13px;font-weight:700;color:#e50914;margin-left:8px;}

/* ══════════════════════════════════════════════
   📈  TRENDING SECTION
══════════════════════════════════════════════ */
.trend-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:10px;}
.trend-card{background:#0e0e14;border:1px solid #18181e;border-radius:12px;padding:14px;position:relative;animation:fadeUp .6s ease both;transition:border-color .3s,transform .3s;}
.trend-card:nth-child(1){animation-delay:.05s;}.trend-card:nth-child(2){animation-delay:.12s;}.trend-card:nth-child(3){animation-delay:.19s;}.trend-card:nth-child(4){animation-delay:.26s;}.trend-card:nth-child(5){animation-delay:.33s;}
.trend-card:hover{border-color:rgba(229,9,20,.3);transform:translateY(-4px);}
.trend-rank{position:absolute;top:10px;right:10px;font-size:9px;font-weight:700;color:#e50914;letter-spacing:1px;}
.trend-emoji{font-size:1.8rem;margin-bottom:8px;display:block;}
.trend-title{font-family:'Playfair Display',serif;font-size:12px;font-weight:700;color:#f0ece4;line-height:1.3;margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.trend-author{font-size:9.5px;color:#2e2e3a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
.trend-bar-wrap{display:flex;align-items:center;gap:6px;}
.trend-bar{flex:1;height:2px;background:#14141c;border-radius:10px;overflow:hidden;}
.trend-fill{height:100%;background:linear-gradient(90deg,#e50914,#ff7055);border-radius:10px;animation:ratingIn 1.2s ease both;}
.trend-num{font-size:10px;font-weight:700;color:#e50914;}

/* ══════════════════════════════════════════════
   🧠  MOOD SECTION
══════════════════════════════════════════════ */
.mood-picker{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;}
.mood-tile{background:#0e0e14;border:1px solid #18181e;border-radius:14px;padding:20px 14px;text-align:center;transition:all .3s ease;animation:fadeUp .5s ease both;}
.mood-tile:nth-child(1){animation-delay:.05s;}.mood-tile:nth-child(2){animation-delay:.12s;}.mood-tile:nth-child(3){animation-delay:.19s;}.mood-tile:nth-child(4){animation-delay:.26s;}.mood-tile:nth-child(5){animation-delay:.33s;}.mood-tile:nth-child(6){animation-delay:.40s;}.mood-tile:nth-child(7){animation-delay:.47s;}.mood-tile:nth-child(8){animation-delay:.54s;}
.mood-tile:hover{border-color:rgba(229,9,20,.35);transform:translateY(-5px);box-shadow:0 14px 30px rgba(229,9,20,.1);}
.mood-tile.selected{border-color:#e50914;background:rgba(229,9,20,.07);}
.mood-tile-icon{font-size:2.2rem;display:block;margin-bottom:8px;}
.mood-tile-label{font-size:11px;font-weight:700;color:#f0ece4;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:3px;}
.mood-tile-sub{font-size:9.5px;color:#333;letter-spacing:.5px;}

/* ══════════════════════════════════════════════
   FUTURE SCOPE
══════════════════════════════════════════════ */
.fs-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:8px;}
.fs-card{background:#0e0e14;border:1px solid #18181e;border-radius:16px;padding:24px 22px;position:relative;overflow:hidden;opacity:0;animation:fadeUp .7s cubic-bezier(.22,1,.36,1) both;transition:border-color .3s,transform .3s,box-shadow .3s;}
.fs-card:nth-child(1){animation-delay:.08s;}.fs-card:nth-child(2){animation-delay:.18s;}.fs-card:nth-child(3){animation-delay:.28s;}.fs-card:nth-child(4){animation-delay:.38s;}
.fs-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(to bottom,#e50914,rgba(229,9,20,0));border-radius:16px 0 0 16px;}
.fs-card::after{content:'';position:absolute;top:-60px;right:-60px;width:160px;height:160px;background:radial-gradient(circle,rgba(229,9,20,.06),transparent 70%);border-radius:50%;pointer-events:none;}
.fs-card:hover{border-color:rgba(229,9,20,.3);transform:translateY(-5px);box-shadow:0 20px 40px rgba(229,9,20,.1);}
.fs-icon{font-size:2rem;margin-bottom:12px;display:block;}
.fs-tag{font-size:8.5px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#e50914;background:rgba(229,9,20,.09);border:1px solid rgba(229,9,20,.15);border-radius:20px;padding:3px 10px;display:inline-block;margin-bottom:10px;}
.fs-card-title{font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;color:#f0ece4;margin-bottom:8px;line-height:1.3;}
.fs-card-desc{font-size:12.5px;color:#484855;line-height:1.7;}
.fs-conclusion{background:linear-gradient(135deg,#0e0e14,#12101a);border:1px solid #1e1e28;border-radius:16px;padding:28px 30px;margin-top:16px;position:relative;overflow:hidden;text-align:center;opacity:0;animation:fadeUp .7s .5s cubic-bezier(.22,1,.36,1) both;}
.fs-conclusion::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 50% 120%,rgba(229,9,20,.07),transparent 65%);pointer-events:none;}
.fs-conclusion-quote{font-family:'Playfair Display',serif;font-size:1rem;font-style:italic;color:#555;line-height:1.8;max-width:720px;margin:0 auto 16px;}
.fs-conclusion-quote em{color:#e50914;font-style:normal;}
.fs-divider{width:0;height:1px;background:linear-gradient(90deg,transparent,#e50914,transparent);margin:16px auto;animation:re 1.2s 1s ease forwards;}
@keyframes re{to{width:160px;}}
.fs-sub{font-size:9.5px;letter-spacing:4px;color:#282832;text-transform:uppercase;}
.fs-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(229,9,20,.08);border:1px solid rgba(229,9,20,.18);border-radius:20px;padding:5px 14px;font-size:9.5px;letter-spacing:2px;text-transform:uppercase;color:#e50914;margin-bottom:18px;}
.fs-pulse{width:5px;height:5px;background:#e50914;border-radius:50%;animation:pp 1.4s ease-in-out infinite;}
@keyframes pp{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(.5);}}

/* ── FOOTER ── */
.footer{text-align:center;padding:2.5rem 1rem 1.5rem;font-size:.7rem;color:#1e1e26;letter-spacing:3px;text-transform:uppercase;border-top:1px solid #101016;margin-top:3rem;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  CURSOR + PARTICLES
# ═══════════════════════════════════════════════════════════
components.html("""
<style>body{margin:0;overflow:hidden;background:transparent;}</style>
<canvas id="pc" style="position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.28;"></canvas>
<script>
(function(){
  var cv=document.getElementById('pc'),ctx=cv.getContext('2d');
  function rsz(){cv.width=window.innerWidth;cv.height=window.innerHeight;}
  rsz();window.addEventListener('resize',rsz);
  var pts=Array.from({length:75},function(){return{x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,r:Math.random()*1.5+.3,dx:(Math.random()-.5)*.22,dy:-(Math.random()*.32+.07),a:Math.random()*.5+.1,red:Math.random()>.62};});
  function pf(){ctx.clearRect(0,0,cv.width,cv.height);pts.forEach(function(p){ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=p.red?'#e50914':'#fff';ctx.globalAlpha=p.a;ctx.fill();p.x+=p.dx;p.y+=p.dy;p.a-=.00035;if(p.y<-5||p.a<=0){p.x=Math.random()*cv.width;p.y=cv.height+5;p.a=Math.random()*.5+.1;}if(p.x<0||p.x>cv.width)p.dx*=-1;});ctx.globalAlpha=1;requestAnimationFrame(pf);}
  pf();
  var pd=window.parent.document,pw=window.parent;
  ['__crs','__cro','__cri','__crd','__crtc'].forEach(function(id){var e=pd.getElementById(id);if(e)e.parentNode.removeChild(e);});
  var s=pd.createElement('style');s.id='__crs';
  s.textContent='*,body,a,button,select,input,textarea,label,[role="button"]{cursor:none!important;}'
    +'#__cro{position:fixed;border-radius:50%;border:1.5px solid rgba(229,9,20,.7);pointer-events:none;z-index:999999;transform:translate(-50%,-50%);width:42px;height:42px;transition:width .4s cubic-bezier(.22,1,.36,1),height .4s cubic-bezier(.22,1,.36,1),border-color .35s,background .35s;}'
    +'#__cro.hov{width:62px!important;height:62px!important;border-color:#e50914;background:rgba(229,9,20,.07);}'
    +'#__cro.clk{width:22px!important;height:22px!important;background:rgba(229,9,20,.22);}'
    +'#__cri{position:fixed;border-radius:50%;border:1px solid rgba(229,9,20,.2);pointer-events:none;z-index:999999;transform:translate(-50%,-50%);width:64px;height:64px;transition:width .55s cubic-bezier(.22,1,.36,1),height .55s cubic-bezier(.22,1,.36,1);}'
    +'#__cri.hov{width:88px!important;height:88px!important;}'
    +'#__crd{position:fixed;width:7px;height:7px;background:#e50914;border-radius:50%;pointer-events:none;z-index:999999;transform:translate(-50%,-50%);box-shadow:0 0 10px 3px rgba(229,9,20,.8);transition:width .12s,height .12s;}'
    +'#__crd.clk{width:14px;height:14px;box-shadow:0 0 22px 8px rgba(229,9,20,.9);}'
    +'#__crtc{position:fixed;inset:0;pointer-events:none;z-index:999998;}';
  pd.head.appendChild(s);
  var ro=pd.createElement('div');ro.id='__cro';ro.style.left='-100px';ro.style.top='-100px';pd.body.appendChild(ro);
  var ri=pd.createElement('div');ri.id='__cri';ri.style.left='-100px';ri.style.top='-100px';pd.body.appendChild(ri);
  var dot=pd.createElement('div');dot.id='__crd';dot.style.left='-100px';dot.style.top='-100px';pd.body.appendChild(dot);
  var tc=pd.createElement('canvas');tc.id='__crtc';pd.body.appendChild(tc);
  var tctx=tc.getContext('2d');
  function trsz(){tc.width=pw.innerWidth;tc.height=pw.innerHeight;}trsz();pw.addEventListener('resize',trsz);
  var mx=pw.innerWidth/2,my=pw.innerHeight/2,r1x=mx,r1y=my,r2x=mx,r2y=my,trail=[];
  pw.addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;dot.style.left=mx+'px';dot.style.top=my+'px';for(var i=0;i<3;i++){trail.push({x:mx+(Math.random()-.5)*5,y:my+(Math.random()-.5)*5,vx:(Math.random()-.5)*2.5,vy:(Math.random()-.5)*2.5-1,r:Math.random()*3.5+2,a:.85+Math.random()*.15,red:Math.random()>.48});}});
  var csel='a,button,input,select,textarea,label,[role="button"],[data-baseweb="tab"]';
  pd.addEventListener('mouseover',function(e){if(e.target.closest(csel)){ro.classList.add('hov');ri.classList.add('hov');}});
  pd.addEventListener('mouseout',function(e){if(e.target.closest(csel)){ro.classList.remove('hov');ri.classList.remove('hov');}});
  pd.addEventListener('mousedown',function(){ro.classList.add('clk');dot.classList.add('clk');for(var i=0;i<24;i++){var ang=(Math.PI*2/24)*i;trail.push({x:mx,y:my,vx:Math.cos(ang)*7*(Math.random()+.35),vy:Math.sin(ang)*7*(Math.random()+.35),r:Math.random()*5+2.5,a:1,red:true});}});
  pd.addEventListener('mouseup',function(){ro.classList.remove('clk');dot.classList.remove('clk');});
  function cframe(){
    r1x+=(mx-r1x)*.09;r1y+=(my-r1y)*.09;ro.style.left=r1x+'px';ro.style.top=r1y+'px';
    r2x+=(mx-r2x)*.16;r2y+=(my-r2y)*.16;ri.style.left=r2x+'px';ri.style.top=r2y+'px';
    tctx.clearRect(0,0,tc.width,tc.height);
    trail.forEach(function(p){p.x+=p.vx;p.y+=p.vy;p.vy+=.07;p.a-=.03;p.r*=.963;if(p.a<=0)return;var g=tctx.createRadialGradient(p.x,p.y,0,p.x,p.y,Math.max(p.r*2.2,.1));g.addColorStop(0,p.red?'rgba(229,9,20,'+p.a+')':'rgba(255,255,255,'+p.a+')');g.addColorStop(1,'rgba(0,0,0,0)');tctx.beginPath();tctx.arc(p.x,p.y,Math.max(p.r*2.2,.1),0,Math.PI*2);tctx.fillStyle=g;tctx.fill();});
    trail=trail.filter(function(p){return p.a>0;});requestAnimationFrame(cframe);}
  cframe();
})();
</script>
""", height=0)

# ═══════════════════════════════════════════════════════════
#  ANIMATED HERO
# ═══════════════════════════════════════════════════════════
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,900;1,400&family=DM+Sans:wght@400;600&display=swap');
.hw{text-align:center;padding:3.5rem 1rem 1.8rem;}
.hey{font-family:'DM Sans',sans-serif;font-size:10px;letter-spacing:6px;color:#e50914;text-transform:uppercase;opacity:0;animation:hfu .5s .1s ease both;margin-bottom:14px;display:flex;align-items:center;justify-content:center;gap:8px;}
.pd{width:5px;height:5px;background:#e50914;border-radius:50%;animation:pp 1.4s ease-in-out infinite;}
@keyframes pp{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(.5);}}
.ht{font-family:'Playfair Display',serif;font-size:clamp(2.2rem,4.5vw,3.8rem);font-weight:900;letter-spacing:-2px;line-height:1.05;margin-bottom:10px;color:#f0ece4;}
.ch{display:inline-block;opacity:0;transform:translateY(-55px) rotateX(80deg);animation:cd .65s cubic-bezier(.22,1,.36,1) both;transform-origin:top center;}
.rd{color:#e50914;font-style:italic;}
.hr{width:0;height:1px;background:linear-gradient(90deg,transparent,#e50914 50%,transparent);margin:14px auto;animation:re 1s 1.3s ease forwards;}
@keyframes re{to{width:200px;}}
.hs{font-family:'DM Sans',sans-serif;font-size:11px;letter-spacing:5px;color:#282832;text-transform:uppercase;opacity:0;animation:hfu .6s 1.5s ease both;}
@keyframes hfu{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
@keyframes cd{to{opacity:1;transform:translateY(0) rotateX(0);}}
</style>
<div class="hw">
  <div class="hey"><div class="pd"></div>AI Powered · Cosine Similarity<div class="pd"></div></div>
  <div class="ht" id="ht"></div>
  <div class="hr"></div>
  <div class="hs">70 curated books &nbsp;·&nbsp; instant matching &nbsp;·&nbsp; live chat</div>
</div>
<script>
(function(){var el=document.getElementById('ht');var words=[{t:'📚 AI Book ',r:false},{t:'Recommender',r:true}];var d=0;words.forEach(function(w){var sp=document.createElement('span');sp.style.display='inline';w.t.split('').forEach(function(ch){var c=document.createElement('span');c.className='ch'+(w.r?' rd':'');c.textContent=ch===' '?'\u00a0':ch;c.style.animationDelay=d+'s';d+=ch===' '?.03:.055;sp.appendChild(c);});el.appendChild(sp);});})();
</script>
""", height=155)

# ═══════════════════════════════════════════════════════════
#  DATA + MODEL
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    df.fillna('', inplace=True)
    df.columns = df.columns.str.strip()
    df['combined_features'] = df['title']*3 + " " + df['genre']*2 + " " + df['author'] + " " + df['description']
    return df.sort_values('rating', ascending=False).reset_index(drop=True)

df = load_data()

@st.cache_data
def init_model(df):
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    fv  = vec.fit_transform(df['combined_features'])
    return vec, fv, cosine_similarity(fv)

vectorizer, feature_vectors, similarity = init_model(df)

def recommend(book, n=6):
    if book not in df['title'].values: return [], []
    idx = df[df['title'] == book].index[0]
    scored = sorted([(i, s*.7+(df.iloc[i]['rating']/5)*.3) for i,s in enumerate(similarity[idx])],key=lambda x:x[1],reverse=True)[1:n+1]
    return [df.iloc[i]['title'] for i,_ in scored], [str(df.iloc[i].get('image_url','')) for i,_ in scored]

# ── Gemini AI chatbot ──────────────────────────────────────
def gemini_response(prompt, api_key):
    """Call Gemini API and return text, or fall back to rule-based."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        book_list = df[['title','author','genre','rating']].to_string(index=False)
        system_ctx = (
            "You are BookBot, an expert AI book recommender. "
            "You have access to this library of 70 books:\n"
            f"{book_list}\n\n"
            "Answer the user's question helpfully. When recommending books, "
            "always pick from the library above. Be concise, friendly, and use emojis."
        )
        response = model.generate_content(f"{system_ctx}\n\nUser: {prompt}")
        return response.text
    except Exception as e:
        return fallback_chatbot(prompt)

def fallback_chatbot(text):
    t = text.lower()
    genre_map = {
        'fantasy':['fantasy','magic','dragon'],'scifi':['sci-fi','space','future','robot'],
        'romance':['love','romance'],'thriller':['thriller','mystery','crime'],
        'horror':['horror','scary','ghost'],'historical':['history','historical','war'],
    }
    for genre, kws in genre_map.items():
        if any(k in t for k in kws):
            m = df[df['genre'].str.contains(genre,case=False,na=False)]
            if not m.empty:
                books = m.nlargest(5,'rating')['title'].tolist()
                return "📚 Here are my top picks:\n\n" + "\n".join([f"**{i+1}.** {b}" for i,b in enumerate(books)])
    for title in df['title']:
        if title.lower() in t:
            b,_ = recommend(title)
            return "📚 Similar books:\n\n" + "\n".join([f"**{i+1}.** {x}" for i,x in enumerate(b[:3])])
    top = df.nlargest(5,'rating')['title'].tolist()
    return "📚 Top rated books:\n\n" + "\n".join([f"**{i+1}.** {b}" for i,b in enumerate(top)])

def show_confetti():
    components.html("""<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
    <script>confetti({particleCount:160,spread:75,origin:{y:.55},colors:['#e50914','#ff6b6b','#fff','#ff4500']});
    setTimeout(function(){confetti({particleCount:60,spread:120,origin:{y:.4},colors:['#e50914','#fff']});},400);</script>""", height=0)

# ── Helpers ────────────────────────────────────────────────
GENRE_EMOJI = {
    'fantasy':'🔮','scifi':'🚀','science fiction':'🚀','romance':'💕','thriller':'🔪',
    'mystery':'🕵️','horror':'👻','historical':'⚔️','biography':'📖','self-help':'💡',
    'philosophy':'🧠','adventure':'🌍','classic':'🎭','dystopian':'🌑',
    'fiction':'📚','business':'💼','finance':'💰','history':'📜','self help':'💡',
}
MOOD_MAP = {
    'Happy 😊':       {'genres':['romance','fiction','comedy'],'desc':'Feel-good, uplifting reads'},
    'Motivational 💪':{'genres':['self-help','biography','business'],'desc':'Books that drive you forward'},
    'Adventure 🌍':   {'genres':['adventure','scifi','fantasy'],'desc':'Epic journeys & new worlds'},
    'Dark & Thrilling 🔪':{'genres':['thriller','mystery','horror'],'desc':'Edge-of-your-seat tension'},
    'Romantic 💕':    {'genres':['romance','fiction'],'desc':'Love stories & heartfelt tales'},
    'Mind-Expanding 🧠':{'genres':['philosophy','science','history'],'desc':'Broaden your perspective'},
    'Scary Night 👻': {'genres':['horror','thriller'],'desc':'Spine-chilling reads'},
    'Relaxing 🌙':    {'genres':['fiction','classic','biography'],'desc':'Easy, comforting reads'},
}

def g_emoji(genre):
    gl = genre.lower()
    for k,v in GENRE_EMOJI.items():
        if k in gl: return v
    return '📚'

def make_cover(poster, title, genre):
    emoji = g_emoji(genre)
    safe  = title.replace('"','&quot;').replace('<','&lt;').replace('>','&gt;')
    spine = (f'<div class="bcover-spine"><div class="bcover-spine-icon">{emoji}</div>'
             f'<div class="bcover-spine-line"></div>'
             f'<div class="bcover-spine-title">{safe}</div></div>')
    if poster and str(poster).strip().startswith('http'):
        src = str(poster).replace('"','%22')
        img = (f'<img src="{src}" alt="{safe}" style="width:100%;height:100%;object-fit:cover;display:block;background:#0d0d16;" '
               f'onerror="this.style.display=\'none\';this.parentNode.querySelector(\'.bcover-spine\').style.display=\'flex\';">'
               + spine.replace('class="bcover-spine"','class="bcover-spine" style="display:none;"'))
    else:
        img = spine
    return f'<div class="bcover">{img}<div class="bcover-shimmer"></div><div class="bcover-shadow"></div></div>'

def build_cards_html(books, posters, prefix=""):
    parts = ['<div class="cards-grid">']
    for book, poster in zip(books, posters):
        rows = df[df['title']==book]
        if rows.empty: continue
        row    = rows.iloc[0]
        rating = float(row['rating'])
        pct    = int(rating/5*100)
        genre  = str(row.get('genre',''))
        author = str(row.get('author',''))
        sb = book.replace('<','&lt;').replace('>','&gt;')
        sa = author.replace('<','&lt;').replace('>','&gt;')
        sg = genre.replace('<','&lt;').replace('>','&gt;')
        cover = make_cover(poster, book, genre)
        parts.append(
            f'<div class="bcard">{cover}'
            f'<div class="bcardBody">'
            f'<div class="bcard-title">{sb}</div>'
            f'<div class="bcard-author">{sa}</div>'
            f'<div class="bcard-genre">{sg}</div>'
            f'<div class="bcard-rating">'
            f'<div class="brating-track"><div class="brating-fill" style="width:{pct}%;"></div></div>'
            f'<span class="brating-num">{rating:.1f}</span>'
            f'</div></div></div>'
        )
    parts.append('</div>')
    return ''.join(parts)

CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}body{background:transparent;padding:4px 0 12px;}
.cards-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;}
.bcard{background:#0e0e14;border:1px solid #18181e;border-radius:14px;overflow:hidden;position:relative;opacity:0;animation:cardIn .75s cubic-bezier(.22,1,.36,1) both;transition:transform .32s,border-color .32s,box-shadow .32s;}
.bcard:nth-child(1){animation-delay:.06s;}.bcard:nth-child(2){animation-delay:.14s;}.bcard:nth-child(3){animation-delay:.22s;}.bcard:nth-child(4){animation-delay:.30s;}.bcard:nth-child(5){animation-delay:.38s;}.bcard:nth-child(6){animation-delay:.46s;}
@keyframes cardIn{from{opacity:0;transform:perspective(520px) rotateX(18deg) translateY(28px) scale(.95);}to{opacity:1;transform:perspective(520px) rotateX(0) translateY(0) scale(1);}}
.bcard::before{content:'';position:absolute;inset:0;border-radius:14px;background:linear-gradient(130deg,rgba(229,9,20,.2),transparent 55%);opacity:0;transition:opacity .35s;z-index:1;pointer-events:none;}
.bcard:hover::before{opacity:1;}.bcard:hover{border-color:rgba(229,9,20,.38);transform:translateY(-11px) scale(1.025);box-shadow:0 0 0 1px rgba(229,9,20,.18),0 24px 48px rgba(229,9,20,.15);}
.bcover{aspect-ratio:2/3;background:linear-gradient(145deg,#0d0d16,#07070e);display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;}
.bcover img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .65s;background:#0d0d16;}
.bcard:hover .bcover img{transform:scale(1.09);}
.bcover-spine{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:16px;background:linear-gradient(160deg,#111120,#080810);}
.bcover-spine-title{font-family:'Playfair Display',serif;font-size:11px;font-weight:700;color:#f0ece4;text-align:center;line-height:1.4;opacity:.8;}
.bcover-spine-line{width:24px;height:1px;background:linear-gradient(90deg,transparent,#e50914,transparent);}
.bcover-spine-icon{font-size:28px;opacity:.6;}
.bcover-shimmer{position:absolute;inset:0;background:linear-gradient(108deg,transparent 38%,rgba(255,255,255,.04) 50%,transparent 62%);background-size:250% 100%;animation:shimmer 2.8s ease-in-out infinite;}
@keyframes shimmer{0%{background-position:-250% 0;}100%{background-position:250% 0;}}
.bcover-shadow{position:absolute;bottom:0;left:0;right:0;height:60%;background:linear-gradient(to top,#0e0e14,transparent);}
.bcardBody{padding:11px;position:relative;z-index:2;}
.bcard-title{font-family:'Playfair Display',serif;font-size:13px;font-weight:700;color:#f0ece4;line-height:1.3;margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.bcard-author{font-size:10px;color:#2e2e3a;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px;}
.bcard-genre{display:inline-block;font-size:9.5px;font-weight:700;color:#e50914;background:rgba(229,9,20,.09);border:1px solid rgba(229,9,20,.18);border-radius:20px;padding:2px 9px;text-transform:uppercase;letter-spacing:1px;margin-bottom:9px;}
.bcard-rating{display:flex;align-items:center;gap:7px;}
.brating-track{flex:1;height:2px;background:#14141c;border-radius:10px;overflow:hidden;}
.brating-fill{height:100%;border-radius:10px;background:linear-gradient(90deg,#e50914,#ff7055);animation:ratingIn 1.3s cubic-bezier(.22,1,.36,1) both;}
.bcard:nth-child(1) .brating-fill{animation-delay:.55s;}.bcard:nth-child(2) .brating-fill{animation-delay:.63s;}.bcard:nth-child(3) .brating-fill{animation-delay:.71s;}.bcard:nth-child(4) .brating-fill{animation-delay:.79s;}.bcard:nth-child(5) .brating-fill{animation-delay:.87s;}.bcard:nth-child(6) .brating-fill{animation-delay:.95s;}
@keyframes ratingIn{from{width:0;}}
.brating-num{font-size:11px;font-weight:700;color:#e50914;min-width:24px;text-align:right;}
</style>
"""

# ── Session state init ─────────────────────────────────────
for key in ['rec_books','rec_posters','favorites','messages']:
    if key not in st.session_state:
        st.session_state[key] = [] if key != 'favorites' else {}

# ═══════════════════════════════════════════════════════════
#  TABS  (7 tabs now)
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Recommend",
    "🧠 Mood Pick",
    "📈 Trending",
    "❤️ Favourites",
    "🔍 Search",
    "🤖 AI Chat",
    "🔭 Future Scope",
])

# ══════════════════════════════════════════════════════════
#  TAB 1 — RECOMMEND
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="sec-title">Pick a book, get <em>5 perfect matches</em></p>', unsafe_allow_html=True)

    # Book of the Day
    botd = df.sample(1, random_state=int(time.strftime("%j"))).iloc[0]
    botd_src = str(botd.get('image_url',''))
    botd_emoji = g_emoji(str(botd.get('genre','')))
    botd_img = (f'<img src="{botd_src.replace(chr(34),"%22")}" style="width:100%;height:100%;object-fit:cover;border-radius:8px;background:#0d0d16;" onerror="this.style.display=\'none\'">'
                if botd_src.strip().startswith('http') else f'<span style="font-size:32px">{botd_emoji}</span>')
    botd_stars = '★'*int(round(float(botd['rating']))) + '☆'*(5-int(round(float(botd['rating']))))
    botd_desc  = str(botd.get('description',''))
    botd_desc  = (botd_desc[:160]+'…') if len(botd_desc)>160 else (botd_desc or f"A must-read in the {botd.get('genre','')} genre.")
    st.markdown(f"""
    <div class="botd">
      <div class="botd-cover">{botd_img}</div>
      <div style="flex:1;">
        <div class="botd-title">{botd['title']}</div>
        <div class="botd-author">{botd.get('author','')}</div>
        <div class="botd-desc">{botd_desc}</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="botd-stars">{botd_stars}</span>
          <span class="botd-num">&nbsp;{float(botd['rating']):.1f}/5</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Quick stats
    st.markdown(f"""<div class="qstrip">
      <div class="qchip">📚 Total <span>{len(df)}</span></div>
      <div class="qchip">🎭 Genres <span>{df['genre'].nunique()}</span></div>
      <div class="qchip">✍️ Authors <span>{df['author'].nunique()}</span></div>
      <div class="qchip">⭐ Top Rated <span>{df['rating'].max():.1f}</span></div>
      <div class="qchip">📊 Avg Rating <span>{df['rating'].mean():.2f}</span></div>
    </div>""", unsafe_allow_html=True)

    # Selector + button
    c_sel, c_btn = st.columns([4,1])
    with c_sel:
        selected = st.selectbox("Book", df['title'].values, label_visibility="collapsed")
    with c_btn:
        clicked = st.button("🚀 Recommend")

    if clicked:
        with st.spinner("Analysing similarity…"):
            time.sleep(0.5)
            books, posters = recommend(selected)
        st.session_state['rec_books']   = books
        st.session_state['rec_posters'] = posters
        if books:
            show_confetti()
            st.success(f"✅ **{len(books)} matches** found for *{selected}*")

    # Book details expander
    if st.session_state.get('rec_books'):
        components.html(CARD_CSS + build_cards_html(st.session_state['rec_books'], st.session_state['rec_posters']), height=540)

        st.markdown("---")
        st.markdown('<p class="sec-title" style="font-size:1.1rem;">📖 <em>Book Details</em></p>', unsafe_allow_html=True)
        detail_pick = st.selectbox("View full details for", ["— select —"] + st.session_state['rec_books'], key="detail_sel")

        if detail_pick != "— select —":
            row    = df[df['title']==detail_pick].iloc[0]
            rating = float(row['rating'])
            stars  = '★'*int(round(rating)) + '☆'*(5-int(round(rating)))
            poster = str(row.get('image_url',''))
            genre  = str(row.get('genre',''))
            author = str(row.get('author',''))
            desc   = str(row.get('description','')) or "No description available."
            pub    = str(row.get('year', row.get('publication_year','')))
            emoji  = g_emoji(genre)
            if poster.strip().startswith('http'):
                cov_html = f'<img src="{poster}" style="width:100%;height:100%;object-fit:cover;background:#0d0d16;" onerror="this.style.display=\'none\'">'
            else:
                cov_html = f'<span style="font-size:3rem">{emoji}</span>'

            col1, col2 = st.columns([1,3])
            with col1:
                st.markdown(f'<div class="detail-box" style="display:block;padding:0;border:none;background:transparent;"><div class="detail-cover">{cov_html}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="padding:8px 0;">
                  <div class="detail-title">{detail_pick}</div>
                  <div class="detail-author">{author}</div>
                  <div class="detail-tags">
                    <span class="detail-tag">{genre}</span>
                    {f'<span class="detail-tag">{pub}</span>' if pub else ''}
                    <span class="detail-tag">Rating {rating:.1f}/5</span>
                  </div>
                  <div class="detail-desc">{desc}</div>
                  <div style="display:flex;align-items:center;gap:6px;">
                    <span class="detail-stars">{stars}</span>
                    <span class="detail-rating-num">{rating:.1f} / 5.0</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Add to favourites button
            fav_key = detail_pick
            already = fav_key in st.session_state['favorites']
            btn_label = "✅ In Favourites" if already else "❤️ Add to Favourites"
            if st.button(btn_label, key="fav_btn"):
                if not already:
                    st.session_state['favorites'][fav_key] = {
                        'title':  detail_pick,
                        'author': author,
                        'genre':  genre,
                        'rating': rating,
                        'poster': poster,
                    }
                    st.success(f"❤️ *{detail_pick}* added to your Favourites!")
                else:
                    del st.session_state['favorites'][fav_key]
                    st.info(f"Removed *{detail_pick}* from Favourites.")

# ══════════════════════════════════════════════════════════
#  TAB 2 — MOOD PICK  🧠
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec-title">What\'s your <em>mood today?</em></p>', unsafe_allow_html=True)
    st.caption("Select a mood and we'll pick the perfect books for how you're feeling right now.")

    # Mood tile grid
    mood_keys  = list(MOOD_MAP.keys())
    mood_cols  = st.columns(4)
    selected_mood = st.session_state.get('selected_mood', None)

    for i, mood in enumerate(mood_keys):
        with mood_cols[i % 4]:
            info = MOOD_MAP[mood]
            is_sel = selected_mood == mood
            border = "border:1px solid #e50914;background:rgba(229,9,20,.07);" if is_sel else "border:1px solid #18181e;"
            icon_part = mood.split(' ')[-1]
            label_part = ' '.join(mood.split(' ')[:-1])
            st.markdown(f"""
            <div class="mood-tile {'selected' if is_sel else ''}" style="{border}margin-bottom:0;">
              <span class="mood-tile-icon">{icon_part}</span>
              <div class="mood-tile-label">{label_part}</div>
              <div class="mood-tile-sub">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Pick", key=f"mood_{i}"):
                st.session_state['selected_mood'] = mood

    if selected_mood:
        st.markdown(f"<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="sec-title" style="font-size:1.1rem;">Books for <em>{selected_mood}</em></p>', unsafe_allow_html=True)
        genres = MOOD_MAP[selected_mood]['genres']
        pattern = '|'.join(genres)
        mood_books = df[df['genre'].str.contains(pattern, case=False, na=False)].nlargest(5,'rating')
        if mood_books.empty:
            mood_books = df.nlargest(5,'rating')
        m_titles  = mood_books['title'].tolist()
        m_posters = [str(r.get('image_url','')) for _,r in mood_books.iterrows()]
        components.html(CARD_CSS + build_cards_html(m_titles, m_posters), height=540)

# ══════════════════════════════════════════════════════════
#  TAB 3 — TRENDING  📈
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-title">📈 <em>Trending</em> & Top Rated</p>', unsafe_allow_html=True)

    # Top 5 overall
    st.markdown("**🏆 All-Time Top 5**")
    top5 = df.nlargest(5,'rating')
    t5_titles  = top5['title'].tolist()
    t5_posters = [str(r.get('image_url','')) for _,r in top5.iterrows()]
    components.html(CARD_CSS + build_cards_html(t5_titles, t5_posters), height=540)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top by genre
    st.markdown('<p class="sec-title" style="font-size:1.1rem;">🎭 Top Rated <em>by Genre</em></p>', unsafe_allow_html=True)
    genre_list = sorted(df['genre'].dropna().unique().tolist())
    sel_genre  = st.selectbox("Choose genre", genre_list, key="trend_genre")
    genre_top  = df[df['genre']==sel_genre].nlargest(5,'rating')
    if not genre_top.empty:
        gt_titles  = genre_top['title'].tolist()
        gt_posters = [str(r.get('image_url','')) for _,r in genre_top.iterrows()]
        components.html(CARD_CSS + build_cards_html(gt_titles, gt_posters), height=540)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full leaderboard
    st.markdown("**📊 Full Leaderboard — Top 10**")
    top10 = df.nlargest(10,'rating')[['title','author','genre','rating']].reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  TAB 4 — FAVOURITES  ❤️
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec-title">Your <em>Favourites</em></p>', unsafe_allow_html=True)

    favs = st.session_state['favorites']

    if not favs:
        st.markdown("""
        <div class="fav-empty">
          <span class="fav-empty-icon">❤️</span>
          Your wishlist is empty.<br>
          Go to <b>Recommend</b>, pick books, view details, and click <em>Add to Favourites</em>.
        </div>""", unsafe_allow_html=True)
    else:
        st.caption(f"{len(favs)} book(s) saved")
        fav_titles  = [v['title']  for v in favs.values()]
        fav_posters = [v['poster'] for v in favs.values()]
        components.html(CARD_CSS + build_cards_html(fav_titles, fav_posters), height=540)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Favourites"):
            st.session_state['favorites'] = {}
            st.rerun()

# ══════════════════════════════════════════════════════════
#  TAB 5 — SEARCH  🔍
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="sec-title">Browse & <em>Filter</em></p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3,2])
    with c1:
        q = st.text_input("Search title or author", placeholder="e.g. Dune, Tolkien…")
        if q:
            res = df[df['title'].str.contains(q,case=False,na=False)|df['author'].str.contains(q,case=False,na=False)]
            st.caption(f"{len(res)} result(s)")
            st.dataframe(res[['title','author','genre','rating']], use_container_width=True, hide_index=True)
    with c2:
        genres = ["All Genres"]+sorted(df['genre'].dropna().unique().tolist())
        g = st.selectbox("Genre", genres)
        filt = df if g=="All Genres" else df[df['genre']==g]
        filt = filt.sort_values('rating', ascending=False)
        st.caption(f"{len(filt)} books")
        st.dataframe(filt[['title','author','rating']], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
#  TAB 6 — AI CHAT  🤖  (Gemini powered)
# ══════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="sec-title">Chat with <em>BookBot AI</em></p>', unsafe_allow_html=True)

    # Gemini API key input
    with st.expander("⚙️ Gemini API Key (optional — enables real AI responses)", expanded=False):
        api_key = st.text_input("Paste your Gemini API key", type="password", key="gemini_key",
                                help="Get a free key at https://aistudio.google.com/app/apikey")
        if api_key:
            st.success("✅ Gemini AI enabled — your chatbot is now powered by Google's AI!")
        else:
            st.info("No key? The bot still works with smart rule-based responses.")

    st.caption("Try: *'suggest fantasy books'*, *'books like Dune'*, *'I feel sad, what should I read?'*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask BookBot anything about books…"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("BookBot is thinking…"):
                key = st.session_state.get("gemini_key","").strip()
                resp = gemini_response(prompt, key) if key else fallback_chatbot(prompt)

            ph, full = st.empty(), ""
            for ch in resp:
                full += ch
                ph.markdown(full + "▋")
                time.sleep(0.009)
            ph.markdown(full)
        st.session_state.messages.append({"role":"assistant","content":resp})

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════
#  TAB 7 — FUTURE SCOPE  🔭
# ══════════════════════════════════════════════════════════
with tab7:
    st.markdown('<p class="sec-title">Future <em>Scope</em></p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#333;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;">Where this system is headed next</div>
    <div class="fs-grid">
      <div class="fs-card"><span class="fs-icon">🧠</span><div class="fs-tag">Phase 1 · AI Upgrade</div><div class="fs-card-title">Deep Learning Recommendations</div><div class="fs-card-desc">Replace TF-IDF cosine similarity with neural collaborative filtering or transformer-based embeddings, dramatically improving recommendation accuracy and contextual understanding of reading preferences.</div></div>
      <div class="fs-card"><span class="fs-icon">👤</span><div class="fs-tag">Phase 2 · Personalization</div><div class="fs-card-title">User Profiles & Reading History</div><div class="fs-card-desc">Implement a secure login system with individual profiles. Track reading history, favourites, and ratings over time to deliver hyper-personalized recommendations that evolve with each reader's taste.</div></div>
      <div class="fs-card"><span class="fs-icon">🎤</span><div class="fs-tag">Phase 3 · Voice</div><div class="fs-card-title">Voice Assistant Integration</div><div class="fs-card-desc">Add speech recognition so users can say "suggest a thriller" and get instant spoken recommendations back — making the system fully hands-free and accessible.</div></div>
      <div class="fs-card"><span class="fs-icon">📱</span><div class="fs-tag">Phase 4 · Deployment</div><div class="fs-card-title">Mobile App & Global Accessibility</div><div class="fs-card-desc">Package as a native iOS and Android application with offline reading lists, push notifications for new releases, and multi-language support to reach a global audience.</div></div>
    </div>
    <div class="fs-conclusion">
      <div class="fs-badge"><div class="fs-pulse"></div>Project Vision</div>
      <div class="fs-conclusion-quote">"In the future, this system can be enhanced by adding <em>personalized recommendations</em> based on user history and preferences. Advanced <em>Deep Learning techniques</em> and real AI chatbots can be integrated to improve recommendation accuracy. The application can also be deployed as a <em>mobile app</em> for better accessibility and real-world usage."</div>
      <div class="fs-divider"></div>
      <div class="fs-sub">📚 AI Book Recommender &nbsp;·&nbsp; Built with TF-IDF · Streamlit · Python</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">📚 AI Book Recommender &nbsp;·&nbsp; 70 Books &nbsp;·&nbsp; TF-IDF · Gemini AI · Streamlit</div>', unsafe_allow_html=True)