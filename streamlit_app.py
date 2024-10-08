import os
import subprocess
import sys
from dateutil.relativedelta import relativedelta
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
    def __init__(self, S, K, maturity, r, q, sigma, n, N, payoff, option_type='Exotic', Ayear=244, unit=1, title='Option Pricing',seed=0):
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
        self.seed=seed

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
        np.random.seed(self.seed)
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


def create_dual_axis_chart(x, y1, y2, key_points=None, use_dual_axis=True, 
                           chart_title="Dual Axis Chart", 
                           y1_title="Y1 Axis", 
                           y2_title="Y2 Axis", 
                           series1_name="Series 1", 
                           series2_name="Series 2"):

    # 创建图表
    fig = go.Figure()

    # 添加第一条数据线（左侧 y 轴或单一 y 轴）
    fig.add_trace(go.Scatter(
        x=x, y=y1, 
        mode='lines',
        name=series1_name,
        line=dict(color='cyan', width=4),
        marker=dict(size=10, color='cyan', symbol='circle', line=dict(color='white', width=2)),
        fill='tonexty',  # 填充到下一个y值，但不填到 x 轴
        fillcolor='rgba(0,255,255,0.2)',  # 半透明区域填充
        hoverinfo='text',
        hovertext=[f"X: {val}<br>Y: {y}" for val, y in zip(x, y1)]
    ))

    # 添加第二条数据线（如果使用双轴则为右侧 y 轴）
    fig.add_trace(go.Scatter(
        x=x, y=y2, 
        mode='lines',
        name=series2_name,
        line=dict(color='magenta', width=4),
        marker=dict(size=10, color='magenta', symbol='circle', line=dict(color='white', width=2)),
        fill='tonexty',  # 填充到下一个y值，但不填到 x 轴
        fillcolor='rgba(255,0,255,0.2)',  # 半透明区域填充
        hoverinfo='text',
        hovertext=[f"X: {val}<br>Y: {y}" for val, y in zip(x, y2)],
        yaxis="y2" if use_dual_axis else 'y'  # 根据是否启用双轴决定 y 轴
    ))

        # 根据是否使用双轴动态更新布局
    # 动态设置 y1 和 y2 轴的最小值和最大值
    y1_min = min(y1) * 0.95  # y1 的最小值设为实际最小值的 95%
    y1_max = max(y1) * 1.05  # y1 的最大值设为实际最大值的 105%

    y2_min = min(y2) * 0.95  # y2 的最小值设为实际最小值的 95%
    y2_max = max(y2) * 1.05  # y2 的最大值设为实际最大值的 105%

    layout_settings = dict(
        title=chart_title,
        title_font=dict(size=30, color='white', family='Arial Black'),
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='white'),
            tickangle=-45,
            zeroline=False,
        ),
        yaxis=dict(
            title=y1_title,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='white'),
            zeroline=False,
            range=[y1_min, y1_max],  # 自动设置 y1 轴的范围，使其合适
        ),
        plot_bgcolor='rgba(10, 10, 10, 0.95)',
        paper_bgcolor='rgba(10, 10, 10, 0.95)',
        font=dict(color='white'),
        legend=dict(
            font=dict(color='white'),
            orientation='h',
            xanchor='center', x=0.5, y=-0.2,
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='white', borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=100),
        hovermode='x',
        dragmode='zoom',
        autosize=True
    )

    # 如果使用双轴，更新右侧 Y 轴
    if use_dual_axis:
        layout_settings['yaxis2'] = dict(
            title=y2_title,
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(color='white'),
            zeroline=False,
            range=[y2_min, y2_max],  # 自动设置 y2 轴的范围，使其合适
        )

    # 更新布局
    fig.update_layout(**layout_settings)

    # 添加渐进式动画
    fig.update_traces(mode="lines", 
                      marker=dict(size=8, opacity=1),
                      line=dict(width=4),
                      selector=dict(type='scatter'))

    fig.update_layout(
        updatemenus=[dict(type="buttons",
                          showactive=False,
                          buttons=[dict(label="PLAY",
                                        method="animate",
                                        args=[None, dict(frame=dict(duration=500, redraw=True),
                                                         fromcurrent=True,
                                                         mode='immediate')])])]
    )

    # 定义动画帧
    frames = [go.Frame(data=[go.Scatter(x=x[:k+1], y=y1[:k+1],
                                        mode='lines', line=dict(color='cyan')),
                             go.Scatter(x=x[:k+1], y=y2[:k+1],
                                        mode='lines', line=dict(color='magenta'))])
              for k in range(1, len(x))]

    # 添加帧到图表
    fig.frames = frames

    # 如果提供了关键点字典，添加注释
    if key_points:
        for key, details in key_points.items():
            key_date = pd.to_datetime(key, format='%Y%m%d')  # 格式化 key 为 datetime 类型
            if key_date in pd.to_datetime(x).values:
                index = np.where(pd.to_datetime(x) == key_date)[0][0]  # 查找 key 对应的索引
                if details["line"] == "y1":
                    y_value = y1[index]
                elif details["line"] == "y2":
                    y_value = y2[index]
                else:
                    continue  # 如果没有指定正确的线条，跳过此注释
                
                # 使用 ax 和 ay 参数手动调整注释框位置
                fig.add_annotation(
                    x=key_date,
                    y=y_value,
                    text=details["text"],
                    showarrow=True,  # 保持箭头
                    arrowhead=2,
                    ax=details.get("ax", 0),  # 获取自定义的 x 轴方向偏移量，默认为 0
                    ay=details.get("ay", -40),  # 获取自定义的 y 轴方向偏移量，默认为 -40
                    font=dict(color='white', size=12, family='Arial'),
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='magenta',
                    borderwidth=2
                )

    return fig
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
    fig=create_dual_axis_chart(x=df['trade_date'], y1=df['close'], y2=df['oi'], key_points=None, use_dual_axis=True, 
                           chart_title=f"{ts_code} Chart", 
                           y1_title="收盘价", 
                           y2_title="持仓量", 
                           series1_name="收盘价", 
                           series2_name="持仓量")
    return df,fig       


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
    根据期货合约的上市日期和到期日期计算期权可选期限和剩余交易天数（假设每个月有22个交易日）
    :param list_date: 上市日期 (格式: 'YYYYMMDD')
    :param delist_date: 到期日期 (格式: 'YYYYMMDD')
    :return: list 可选择的期权期限 和 到期日前剩余的交易日数
    """
    # 将日期字符串转换为 datetime 对象
    list_date = datetime.datetime.strptime(list_date, '%Y%m%d')
    delist_date = datetime.datetime.strptime(delist_date, '%Y%m%d')

    # 获取当前日期
    today = datetime.datetime.today()

    # 确定合约到期前一个月的15号
    if delist_date.month == 1:
        # 如果到期日在1月，则前一个月的15号为前一年的12月15日
        previous_month_15th = datetime.datetime(delist_date.year - 1, 12, 15)
    else:
        # 否则，计算前一个月的15号
        previous_month_15th = datetime.datetime(delist_date.year, delist_date.month - 1, 15)

    # 计算从今天到前一个月15号之间的天数
    days_difference = abs((previous_month_15th - today).days)

    # 假设每个月有22个交易日，计算剩余的交易天数
    trading_days = (days_difference / 30) * 22  # 按每个月22个交易日估算

    # 计算从当前日期到到期日前一个月15号的剩余月份数
    remaining_months = (previous_month_15th.year - today.year) * 12 + (previous_month_15th.month - today.month)

    # 生成可选的期权期限列表（单位：月份）
    if trading_days >= remaining_months*22:
        maturities = [f'{i}个月' for i in range(1, remaining_months + 1)]
    else:
        maturities = [f'{i}个月' for i in range(1, remaining_months)]
        
    return maturities, round(trading_days)


# 根据当前日期匹配指定交易日天数后的日期
def get_future_trading_date(trading_days_after=22):
    """
    根据当前日期，返回指定交易日数量之后的日期
    :param trading_days_after: 交易日数，默认为22
    :return: 未来的交易日日期 (格式: 'YYYY-MM-DD')
    """

    # 获取当前日期
    today = datetime.datetime.today()
    if trading_days_after %22 ==0:
        future_trading_date = today+relativedelta(months=trading_days_after//22) # 索引从0开始，取第22个交易日
    else:
        future_trading_date = today+datetime.timedelta(round(trading_days_after/22*30))
    future_trading_date=future_trading_date.strftime('%Y%m%d')

    return future_trading_date


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
    

# Streamlit布局美化

st.set_page_config(page_title="保险+期货报价")



st.title("保险+期货 报价程序")
try:
    st.image('https://pica.zhimg.com/80/v2-5d6b7ebf0e05e4babe05153463fe38fb_1440w.png', caption="")
except:
    pass

st.table(pd.DataFrame({'交易所名称':['郑州商品交易所','上海期货交易所','大连商品交易所','上海证券交易所','深圳证券交易所','中国金融期货交易所'],'交易所代码':['CZCE','SHFE','DCE','SSE','SZSE','CFFEX'],'合约后缀':['.ZCE','.SHF','.DCE','.SH','.SZ','.CFX']}))

col1, col2, col3 = st.columns([40, 40, 60])
with col1:
    st.header("期权信息")
    underlying = st.text_input("标的资产", value='LH2503.DCE')
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
    available_maturities,maxtradingday = calculate_option_maturities(contract_info['list_date'], contract_info['delist_date'])
    maturity = st.multiselect("期限", ["1个月", "2个月", "3个月", "4个月",f"最长期限({maxtradingday}天)"],default=available_maturities[:4]+[f"最长期限({maxtradingday}天)"])
    # 根据所选期限自动计算 maturity_input
    maturity_input_list=[]
    for i in maturity:
        try:
            maturity_input_list.append(int(i.split('个月')[0]) * 22)
        except:
            maturity_input_list.append(maxtradingday)
            
   

with col2:
    st.header("定价参数")
    daily_data,fig = get_futures_daily(underlying, '1999-01-01', '2099-01-04')
            
    if daily_data.empty:
        st.write(f"没有找到 {underlying} 的价格数据。")
    else:
        # 计算历史波动率
        volatility = calculate_historical_volatility(daily_data, 22)
  
    Insurance_scale = st.number_input("保费规模", value=2000000)
    spot_price = st.number_input("入场价格", value=round(daily_data['close'].to_list()[-1],2),step=1.0,max_value=round(daily_data['close'].to_list()[-1]*1.1,2),min_value=round(daily_data['close'].to_list()[-1]*0.9,2))
    strike_ratio = float(st.text_input("行权比率", value='1'))
    strike_price= round(spot_price*strike_ratio,2)
    barrier_price = st.number_input("赔付障碍", value=None,step=0.01)
    volatility1 = st.number_input("定价波动率", value=0.25)
    volatility2 = st.number_input("对冲波动率", round(volatility,2))
    r = st.number_input("无风险利率", value=0.01)
    N = st.number_input("模拟次数", value=10000, step=1000)
    random_seed = st.number_input("随机数种子", value=0)

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
                try:
                    payoff_func = get_payoff_function(option_type, CP, strike_price, barrier_price)
                    S = [spot_price]
                    option = OTCOptionPricingSystem(S=S, K=strike_price, maturity=maturity_input, r=r, q=0, sigma=volatility1, n=10, N=N, payoff=payoff_func, option_type=option_type,seed=random_seed)
                    option.generate_paths()
                    price = option.cal_price()
                    
                    hedge = OTCOptionPricingSystem(S=S, K=strike_price, maturity=maturity_input, r=r, q=0, sigma=volatility2, n=10, N=N, payoff=payoff_func, option_type=option_type,seed=random_seed)
                    hedge.generate_paths()
                    hedge.cal_price()
                    hedge.Greeks(name='delta')
                    
                    #计算吨数、权利金和保费
                    trade_num = round(Insurance_scale * (1 - 0.05 - 0.3) / price, 2)
                    rights_fee = round(trade_num * price + Insurance_scale * 0.3, 2)
                    insurance_fee = round(trade_num * price / 0.65, 2)
                    today=datetime.datetime.today().strftime('%Y%m%d')
                    endday=get_future_trading_date(maturity_input)
                    result_data.append({
                        "期权类型": option_type,
                        "标的代码": underlying,
                        "交易方式": tradeway,
                        "开始日期": f'{today}',
                        "结束日期": f'{endday}',
                        "采价期": f'{today}-{endday}',
                        "入场价格": spot_price,
                        "执行价格": strike_price,
                        "交易数量": trade_num,
                        '权利金': rights_fee,
                        '保费': insurance_fee,
                        '入场手数': int(hedge.delta * trade_num / int(multiplier)),
                        '报价权利金率': round(price / strike_price / 0.65 * 0.95, 4),
                        '实际权利金率': round(price / strike_price , 4),
                        '保费率': round(price / strike_price / 0.65, 4),
                        '估算天数':f'{maturity_input}'
                    })

                    table_data.append({
                        "标的合约": underlying,
                        "期权类型": option_type,
                        "交易方向": "大地保险买入" if tradeway == '海通卖出' else "大地保险卖出",
                        "期权期限": f'{today}-{endday}',
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
                except ValueError:
                    st.write('过于虚值！请检查行权比例')
             
                


        # 转化结果为dataframe
        results_df = pd.DataFrame(result_data)
        st.write('交易指令')
        st.dataframe(results_df.T)
        st.write('Excel报价')
        st.dataframe(table_data)

st.plotly_chart(fig)







