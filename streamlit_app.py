import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import plotly.figure_factory as ff
import pandas as pd
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import datetime
import tushare as ts

ts.set_token('69e781cb04e2be652812ccf4adf976dc2cb8b592d1ceb4c4b6732e22')  # 替换为你的 Tushare API Token
pro = ts.pro_api()

warnings.filterwarnings('ignore')

class OTCOptionPricingSystem:
    def __init__(self, S, K, maturity, r, q, sigma, n, N, payoff, option_type='Exotic', Ayear=244, unit=1, title='Option Pricing'):
        self.S = S
        self.K = K
        self.maturity = maturity
        self.r = r
        self.q = q
        self.sigma = sigma
        self.n = n
        self.N = N
        self.payoff = payoff
        self.option_type = option_type
        self.Ayear = Ayear
        self.unit = unit
        self.title = title
        self.T = self.maturity / self.Ayear  # 年化到期时间
        self.steps = int(self.n * self.maturity)
        self.dt = 1 / self.n / self.Ayear

        self.W = None
        self.geo_paths = None
        self.geo_paths2 = None
        self.payoff_list = None
        self.close_price = None
        
        self.price = None
        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega = None
        self.rho = None
        self.fig = None
        self.traceback_df=None
        self.initial_price = None  # 缓存初始价格

    def generate_paths(self):
        np.random.seed(0)
        norm_matrix = np.random.normal(size=(self.steps, self.N))
        ST = np.log(self.S[-1]) + np.cumsum(
            ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * norm_matrix),
            axis=0)
        self.geo_paths = np.exp(ST)
        self.W = norm_matrix
        self.geo_paths2 = np.exp(
            np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -norm_matrix),
                axis=0))

    def cal_price(self, paths1=None, paths2=None,remain=1):
        if remain==1:
            maturity=self.maturity
        else:
            maturity=self.maturity-1
    
        if paths1 is None:
            paths1 = self.geo_paths
        if paths2 is None:
            paths2 = self.geo_paths2

        if self.maturity != 0:
            close_price = paths1[[(x + 1) * self.n - 1 for x in range(maturity)], :].T
            close_price2 = paths2[[(x + 1) * self.n - 1 for x in range(maturity)], :].T
            payoff_list = [self.payoff(self.S + list(close_price[x])) for x in range(close_price.shape[0])]
            payoff_list2 = [self.payoff(self.S + list(close_price2[x])) for x in range(close_price2.shape[0])]
            price = (np.mean(payoff_list) + np.mean(payoff_list2)) / 2 * np.exp(-self.r * maturity / self.Ayear)
        else:
            payoff_list = self.payoff(self.S)
            price = np.mean(payoff_list)
            close_price = self.S

        # 缓存初始价格
        if self.initial_price is None:
            self.initial_price = price
        
        self.price = price
        self.payoff_list = payoff_list
        self.close_price = close_price
        return price

    def Greeks(self, name):
        # 确保初始价格只计算一次
        if self.initial_price is None:
            self.cal_price()

        if name == 'delta':
            PV_Sup = self.cal_price(
                np.exp(np.log(self.S[-1] * 1.001) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.W),
                    axis=0)),
                np.exp(np.log(self.S[-1] * 1.001) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -self.W),
                    axis=0)))
            PV_Sdown = self.cal_price(
                np.exp(np.log(self.S[-1] * 0.999) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.W),
                    axis=0)),
                np.exp(np.log(self.S[-1] * 0.999) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -self.W),
                    axis=0)))
            self.delta = (PV_Sup - PV_Sdown) / (self.S[-1] * 0.002)

        elif name == 'gamma':
            PV_Sup = self.cal_price(
                np.exp(np.log(self.S[-1] * 1.01) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.W),
                    axis=0)),
                np.exp(np.log(self.S[-1] * 1.01) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -self.W),
                    axis=0)))
            PV_Sdown = self.cal_price(
                np.exp(np.log(self.S[-1] * 0.99) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.W),
                    axis=0)),
                np.exp(np.log(self.S[-1] * 0.99) + np.cumsum(
                    ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -self.W),
                    axis=0)))
            self.gamma = (PV_Sup + PV_Sdown - 2 * self.initial_price) / ((self.S[-1] * 0.01) ** 2)

        elif name == 'theta':
            norm_matrix2 = np.random.normal(size=(self.steps - self.n, self.N))
            geo_path_downT = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * norm_matrix2),
                axis=0))
            geo_path_downT2 = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -norm_matrix2),
                axis=0))
            self.theta = self.cal_price(geo_path_downT, geo_path_downT2,0) - self.initial_price

        elif name == 'vega':
            geo_path_upSigma = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - (self.sigma + 0.005)**2 / 2) * self.dt + (self.sigma + 0.005) * np.sqrt(self.dt) * self.W),
                axis=0))
            geo_path_upSigma2 = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - (self.sigma + 0.005)**2 / 2) * self.dt + (self.sigma + 0.005) * np.sqrt(self.dt) * -self.W),
                axis=0))
            geo_path_downSigma = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - (self.sigma - 0.005)**2 / 2) * self.dt + (self.sigma - 0.005) * np.sqrt(self.dt) * self.W),
                axis=0))
            geo_path_downSigma2 = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r - self.q - (self.sigma - 0.005)**2 / 2) * self.dt + (self.sigma - 0.005) * np.sqrt(self.dt) * -self.W),
                axis=0))
            self.vega = self.cal_price(geo_path_upSigma, geo_path_upSigma2) - self.cal_price(geo_path_downSigma, geo_path_downSigma2)

        elif name == 'rho':
            geo_path_upR = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r + 0.001 - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.W),
                axis=0))
            geo_path_upR2 = np.exp(np.log(self.S[-1]) + np.cumsum(
                ((self.r + 0.001 - self.q - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * -self.W),
                axis=0))
            self.rho = self.cal_price(geo_path_upR, geo_path_upR2) - self.initial_price

    def Plot(self, manual_pv=0):
        # 创建模拟数据跟踪表
        traceback_df = pd.DataFrame(self.close_price)
        traceback_df.columns = ['SmuD' + str(int(i) + 1) for i in traceback_df.columns]
        traceback_df.index = ['Path' + str(int(i) + 1) for i in traceback_df.index]

        Actcount = 0
        for i in self.S:
            Actcount += 1
            traceback_df.insert(loc=Actcount - 1, column='ActD' + str(Actcount), value='')
            traceback_df['ActD' + str(Actcount)] = i

        traceback_df['Payoff'] = self.payoff_list

        # 数据可视化
        fig1 = px.line(self.geo_paths[:, 1:101], labels={'index': 'Time Increments', 'value': 'Price'}, template='simple_white', width=450, height=350)
        annotations = [a.to_plotly_json() for a in fig1["layout"]["annotations"]]
        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.1,
                                xanchor='left', yanchor='bottom',
                                text='Monte Carlo Paths（First 100）',
                                font=dict(family='Arial', size=25),
                                showarrow=False))

        fig1.update_layout(height=450, width=700,
                           showlegend=True,
                           autosize=True, template="plotly_dark", annotations=annotations)

        try:
            # 赔付分布可视化
            hist_data = [[x / self.price for x in self.payoff_list]]
            fig2 = ff.create_distplot(hist_data, group_labels=['pdf'], show_hist=False, show_rug=False)

            annotations = [a.to_plotly_json() for a in fig2["layout"]["annotations"]]
            annotations.append(dict(xref='paper', yref='paper', x=0, y=1.1,
                                    xanchor='left', yanchor='bottom',
                                    text='Payoff Probability Density',
                                    font=dict(family='Arial', size=25),
                                    showarrow=False))

            fig2.update_layout(height=350, width=500,
                               showlegend=False,
                               autosize=True, template="plotly_dark", annotations=annotations)
        except:
            np.random.seed(1)
            x = np.random.randn(5)
            hist_data = [x]
            fig2 = ff.create_distplot(hist_data, group_labels=['pdf'], show_hist=False, show_rug=False)

        # 整合图
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Monte Carlo Paths（First 100）', 'Payoff Probability Density'])
        for trace2 in fig2.select_traces():
            fig.add_trace(trace2, row=1, col=2)

        for trace1 in fig1.select_traces():
            fig.add_trace(trace1, row=1, col=1)

        text1 = f'<br>Parameters<br>Type：{self.option_type}<br>Spot Price：{self.S[-1]}<br>Strike Price：{self.K}<br>Time to maturity：{self.maturity}<br>Risk free interest rate：{self.r}<br>Volatility：{self.sigma}'
        if manual_pv == 0:
            text2 = f'<br>Outputs<br>Price：{self.price * self.unit}<br>Delta：{self.delta * self.unit}<br>Gamma：{self.gamma * self.unit}<br>Theta：{self.theta * self.unit}<br>Vega：{self.vega * self.unit}<br>Rho：{self.rho * self.unit}<br>'
        else:
            text2 = f'<br>Outputs<br>Price：{manual_pv}<br>Delta：{self.delta * self.unit}<br>Gamma：{self.gamma * self.unit}<br>Theta：{self.theta * self.unit}<br>Vega：{self.vega * self.unit}<br>Rho：{self.rho * self.unit}<br>'

        annotations = [a.to_plotly_json() for a in fig["layout"]["annotations"]]
        annotations.append(dict(xref='paper', yref='paper', x=-0.03, y=1.3,
                                xanchor='left', yanchor='bottom',
                                text=self.title,
                                font=dict(family='Arial', size=25),
                                showarrow=False))

        annotations.append(dict(xref='paper', yref='paper', x=0, y=-1.4,
                                align='left',
                                text=text1,
                                font=dict(family='Arial', size=15),
                                showarrow=False, bordercolor='white', borderwidth=1))

        annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-1.3,
                                align='left',
                                text=text2,
                                font=dict(family='Arial', size=15),
                                showarrow=False, bordercolor='white', borderwidth=1))

        annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-1.4,
                                xanchor='center', yanchor='top',
                                text='Coded by Wenhang Gu',
                                font=dict(family='Arial', size=12, color='rgb(150,150,150)'),
                                showarrow=False))

        fig.update_xaxes(showline=True, showgrid=False, linecolor='black', ticks='outside', tickcolor='black')
        fig.update_yaxes(showline=True, showgrid=True, gridcolor='lightgrey', linecolor='black', ticks='outside', tickcolor='black')

        fig.update_layout(height=450, width=700,
                          showlegend=True,
                          autosize=True, annotations=annotations,
                          margin=dict(b=215))

        self.fig = fig
        self.traceback_df=traceback_df


# 获取期货合约基本信息
def get_active_futures_contract_info(exchange=None):
    # 获取所有合约信息
    df = pro.fut_basic(exchange=exchange)
    
    # 当前日期
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    # 过滤仍在上市的合约（delist_date 为空或在今天之后）
    df_active = df[(df['delist_date'].isna()) | (df['delist_date'] > today)]
    
    return df_active

# 获取期货合约的日线行情数据
def get_futures_daily(ts_code, start_date, end_date):
    df = pro.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_index(ascending=False)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df       


def calculate_historical_volatility(daily_data, days):
    # 计算对数收益率
    daily_data['log_return'] = np.log(daily_data['close'] / daily_data['close'].shift(1))
    
    # 去掉缺失值
    daily_data = daily_data.dropna()

    # 使用最近的 'days' 天数据计算波动率
    recent_data = daily_data.tail(days)
    
    # 计算年化波动率
    volatility = np.std(recent_data['log_return']) * np.sqrt(244)  # 252 为一年交易日数
    
    return volatility


def call_payoff(close, k ,B=None):
    if B==None:
        return max(close[-1] - k, 0)
    else:
        return min(max(close[-1] - k, 0),B)  # Call option

def put_payoff(close, k ,B=None):
    if B==None:
        return max(k - close[-1], 0)  # Put option
    else:
        return max(max(k - close[-1], 0),B)

def Asian_call_payoff(close, k ,B=None):
    settle=np.mean(close)
    if B==None:
        return max(settle - k, 0)
    else:
        return min(max(settle - k, 0),B)

def Asian_put_payoff(close, k ,B=None):
    settle=np.mean(close)
    if B==None:
        return max(k - settle, 0)
    else:
        return max(max(k - settle, 0))

def Enhanced_Asian_call_payoff(close, k ,B=None):
    pick_price=[max(x,k) for x in close]
    settle=np.mean(pick_price)
    if B == None:
        return max(settle - k, 0)
    else:
        return min(max(settle - k, 0),0) 

def Enhanced_Asian_put_payoff(close, k ,B=None):
    pick_price=[min(x,k) for x in close]
    settle=np.mean(pick_price)
    if B == None:
        return max(k - settle, 0) 
    else:
        return max(max(k - settle, 0),0)
    

# 获取期货合约的基础信息
def get_futures_contract_info(ts_code):
    """
    获取期货合约的基本信息，包括上市日期和到期日期
    :param ts_code: 期货合约代码
    :return: dict 包含上市日期和到期日期
    """
    if ts_code.split('.')[1]=='ZCE':
        exchange='CZCE'
    else:
        exchange=ts_code.split('.')[1]
        
    df = pro.fut_basic(ts_code=ts_code,exchange=exchange)
    if not df.empty:
        list_date = df.iloc[0]['list_date']
        delist_date = df.iloc[0]['delist_date']
        per_unit= df.iloc[0]['per_unit']
        return {'list_date': list_date, 'delist_date': delist_date ,'per_unit' : per_unit}
    return None

# 根据合约信息计算可选择的期权期限
def calculate_option_maturities(list_date, delist_date):
    """
    根据期货合约的上市日期和到期日期计算期权可选期限
    :param list_date: 上市日期
    :param delist_date: 到期日期
    :return: list 可选择的期权期限
    """
    # 转换为 datetime 对象
    list_date = datetime.datetime.strptime(list_date, '%Y%m%d')
    delist_date = datetime.datetime.strptime(delist_date, '%Y%m%d')
    
    # 计算从当前日期到合约到期的月份数
    current_date = datetime.datetime.now()
    remaining_months = (delist_date.year - current_date.year) * 12 + (delist_date.month - current_date.month)
    
    # 生成可选的期权期限（单位：月份）
    maturities = []
    for i in range(1, remaining_months + 1):
        maturities.append(f'{i}个月')
    
    return maturities




# 初始化期权期限列表
available_maturities = ["1个月", "2个月", "3个月", "4个月"]


# Streamlit布局美化

st.set_page_config(page_title="保险+期货报价")



st.title("保险+期货 报价程序")
try:
    st.image('https://pica.zhimg.com/80/v2-5d6b7ebf0e05e4babe05153463fe38fb_1440w.png', caption="", width=300)
except:
    pass

col1, col2, col3 = st.columns([40, 40, 60])
with col1:
    st.header("期权信息")
    underlying = st.text_input("标的资产", value='LH2411.DCE')
    contract_info = get_futures_contract_info(underlying)
    multiplier = st.text_input("合约乘数", value=int(contract_info['per_unit']))
    tradeway = st.selectbox("交易方式", ('海通卖出', '海通买入'))
    strikeway = st.selectbox("行权方式", ("欧式", '美式'))
    CP = st.selectbox("方向", ("看涨", "看跌"))
    price_picker = st.text_input('输入采价期', value=None)

    # 多选框选择多个期权结构
    options = ["欧式期权", "亚式期权", "亚式增强期权"]
    selected_options = st.multiselect(
        "选择期权结构",
        options,
        default=["欧式期权"]
    )

    # 使用动态更新的可选期限
    maturity = st.multiselect("期限", ["1个月", "2个月", "3个月", "4个月"],default=['1个月'])
    
    # 根据所选期限自动计算 maturity_input
    maturity_input_list = [int(m.split('个月')[0]) * 22 for m in maturity]

   

with col2:
    st.header("定价参数")
    daily_data = get_futures_daily(underlying, '1999-01-01', '2099-01-04')
            
    if daily_data.empty:
        st.write(f"没有找到 {underlying} 的价格数据。")
    else:
        # 计算历史波动率
        volatility = calculate_historical_volatility(daily_data, 22)
  
    Insurance_scale = st.number_input("保费规模", value=2000000)
    spot_price = st.number_input("入场价格", value=round(daily_data['close'].to_list()[-1],2),step=1.0)
    strike_ratio = st.number_input("行权比率", value=1)
    strike_price= round(spot_price*strike_ratio,2)
    barrier_price = st.number_input("赔付障碍", value=None)
    volatility1 = st.number_input("定价波动率", value=0.25)
    volatility2 = st.number_input("对冲波动率", round(volatility,2))
    r = st.number_input("无风险利率", value=0.01)
    N = st.number_input("模拟次数", value=10000, step=1000)

def get_payoff_function(option_type, CP, strike_price, barrier_price):
    if option_type == "欧式期权" and CP == '看涨':
        return lambda x: call_payoff(x, strike_price, barrier_price)
    elif option_type == "欧式期权" and CP == '看跌':
        return lambda x: put_payoff(x, strike_price, barrier_price)
    elif option_type == "亚式期权" and CP == '看涨':
        return lambda x: Asian_call_payoff(x, strike_price, barrier_price)
    elif option_type == "亚式期权" and CP == '看跌':
        return lambda x: Asian_put_payoff(x, strike_price, barrier_price)
    elif option_type == "亚式增强期权" and CP == '看涨':
        return lambda x: Enhanced_Asian_call_payoff(x, strike_price, barrier_price)
    elif option_type == "亚式增强期权" and CP == '看跌':
        return lambda x: Enhanced_Asian_put_payoff(x, strike_price, barrier_price)

with col3:
    st.header("报价")
    run_simulation = st.button("定价")
    
    if run_simulation:
        result_data = []
        table_data = []

        # 显示进度条
        progress_bar = st.progress(0)
        total_iterations = len(selected_options) * len(maturity_input_list)

        current_iteration = 0

        for option_type in selected_options:
            for maturity_input in maturity_input_list:
                
                payoff_func = get_payoff_function(option_type, CP, strike_price, barrier_price)
                S = [spot_price]
                option = OTCOptionPricingSystem(S=S, K=strike_price, maturity=maturity_input, r=r, q=0, sigma=volatility1, n=10, N=N, payoff=payoff_func, option_type=option_type)
                option.generate_paths()
                price = option.cal_price()
                
                hedge = OTCOptionPricingSystem(S=S, K=strike_price, maturity=maturity_input, r=r, q=0, sigma=volatility2, n=10, N=N, payoff=payoff_func, option_type=option_type)
                hedge.generate_paths()
                hedge.cal_price()
                hedge.Greeks(name='delta')
                
                #计算吨数、权利金和保费
                trade_num = round(Insurance_scale * (1 - 0.05 - 0.3) / price, 2)
                rights_fee = round(trade_num * price + Insurance_scale * 0.3, 2)
                insurance_fee = round(trade_num * price / 0.65, 2)
                
                result_data.append({
                    "期权类型": option_type,
                    "标的代码": underlying,
                    "交易方式": tradeway,
                    "行权方式": strikeway,
                    "合约期限": f'{maturity_input // 22}个月',
                    "采价期": price_picker,
                    "入场价格": spot_price,
                    "执行价格": strike_price,
                    "交易数量": trade_num,
                    '权利金': rights_fee,
                    '保费': insurance_fee,
                    '入场手数': int(hedge.delta * trade_num / int(multiplier)),
                    '报价权利金率': round(price / strike_price / 0.65 * 0.95, 4),
                    '实际权利金率': round(price / strike_price , 4),
                    '保费率': round(price / strike_price / 0.65, 4)
                })

                table_data.append({
                    "标的合约": underlying,
                    "期权类型": option_type,
                    "交易方向": "大地保险买入" if tradeway == '海通卖出' else "大地保险卖出",
                    "期权期限": f'{maturity_input // 22}个月',
                    "采价期": price_picker,
                    "行权价格": "入场价格*100%",
                    "封顶赔付价格": barrier_price,
                    "期权费率": round(price / strike_price / 0.65 * 0.95, 4),
                    "交易数量": "目标保费/（保险费率*行权价格）",
                    "保险费率": round(price / strike_price / 0.65, 4),
                    "估算保费": insurance_fee,
                    "估算入场价格": spot_price,
                    "估算行权价格": strike_price,
                    "估算吨数": trade_num
                })

                # 更新进度条
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)

                # 模拟计算的延迟，显示进度条变化
                time.sleep(0.5)  # 模拟延迟，可以移除
                


        # 转化结果为dataframe
        results_df = pd.DataFrame(result_data)
        st.write('交易指令')
        st.dataframe(results_df.T)
        st.write('Excel报价')
        st.dataframe(table_data)
        
        
if contract_info:
    available_maturities = calculate_option_maturities(contract_info['list_date'], contract_info['delist_date'])
    st.write(f"期货合约 {underlying} 的基本信息：上市日期: {contract_info['list_date']}，到期日期: {contract_info['delist_date']}，合约乘数: {contract_info['per_unit']}，可选期权期限: {', '.join(available_maturities)}")

else:
    st.write('未读取到相关信息')




