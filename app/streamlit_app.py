from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from tesla_qa.qa_pipeline import FilingQASystem, parse_years

st.set_page_config(page_title='Tesla Filing QA', layout='wide')
st.title('Tesla 跨年财报智能问答系统')
st.caption('支持 Tesla 2021-2025 年 10-K / 10-Q 跨文档问答、混合检索与证据展示，默认使用 Qwen')

index_dir = Path('data/index')

with st.sidebar:
    st.header('查询设置')
    form_filter = st.selectbox('文档范围', ['所有文档', '仅10-K', '仅10-Q'])
    top_k = st.slider('检索条数', 4, 20, 10)
    show_debug = st.checkbox('显示检索调试信息', value=True)

question = st.text_area(
    '输入问题',
    value='对比2021年至2023年，特斯拉在哪个季度的汽车毛利率最高？当时的宏观背景在管理层讨论中是如何描述的？',
    height=120,
)

if st.button('开始问答', type='primary'):
    if not index_dir.exists():
        st.error('未检测到 data/index，请先运行下载、解析、建索引脚本。')
    else:
        qa = FilingQASystem(index_dir)
        ff = None if form_filter == '所有文档' else form_filter.replace('仅', '')
        years = parse_years(question) or None
        with st.spinner('检索与生成中...'):
            resp = qa.answer(question, form_filter=ff, years=years, top_k=top_k)

        st.subheader('答案')
        st.write(resp.answer)

        st.subheader('引用证据')
        citation_df = pd.DataFrame(resp.citations)
        st.dataframe(citation_df, use_container_width=True)

        if show_debug:
            st.subheader('检索证据详情')
            for i, hit in enumerate(resp.retrieved, start=1):
                with st.expander(f'Evidence {i} | {hit.metadata.get("form")} | {hit.metadata.get("filing_date")} | score={hit.final_score:.3f}'):
                    st.json(hit.metadata)
                    st.write(hit.text)

            st.subheader('推理流程调试')
            st.json(resp.reasoning_trace)
