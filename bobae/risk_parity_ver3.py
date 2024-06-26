import numpy as np
import pandas as pd
import os
import gzip, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
from scipy.optimize import minimize
from get_fn_data import sort_item, get_excel_data

def sort_item(data):

    """
    data를 읽고 item별로 정리
    """

    data.columns = data.columns.map(lambda x: x.split(".")[0])
    sort_data = data.T.reset_index(drop=False).set_index("Item Name").T

    new_data = {}
    for item_name in set(sort_data.columns):

        item_data = sort_data[item_name]
        item_data = item_data.T.set_index("index").T
        item_data.index.name = "date"
        item_data.columns.name = None

        if item_name != "시장구분":
            item_data = item_data.apply(pd.to_numeric)

        if "순매수대금(개인)" in item_name:
            item_name = "개인순매수"
        elif "순매수대금(기관계)" in item_name:
            item_name = "기관순매수"
        elif "순매수대금(등록외국인)" in item_name:
            item_name = "등록외국인순매수"
        elif "순매수대금(외국인계)" in item_name:
            item_name = "외인순매수"

        try:
            concat_data = new_data[item_name]
            concat_data = pd.concat([concat_data, item_data], axis=1)
            new_data.update({item_name: concat_data})
        except:
            new_data[item_name] = item_data
    return new_data
def get_excel_data(date, file_name):
    """ raw 파일 읽기

    Parameters
    ----------------------------
    date : 데이터를 가져온 날짜
    file_name : raw 파일명
    """

    original_path = os.getcwd()
    cnt_ = 0
    print(f'현재경로: {os.getcwd()}')
    while 'data' not in os.listdir(os.getcwd()):
        os.chdir('..')
        print(f'기준경로 변경 ==> {os.getcwd()}')
        if cnt_ > 100:
            print('경로설정확인요망')
            raise FileExistsError
    os.chdir('./data')
    print(f'경로변경완료: {os.getcwd()}')

    if not os.path.exists(f'./data/{date}/items'):
        print('##################### 데이터 정리하기 선택')
        file_path = os.path.join(os.path.join(os.getcwd(), f"./{date}"),file_name)            
        data = pd.read_excel(file_path,
                        index_col=0,
                        header=8,
                        thousands=",",
                        skiprows=[9, 10, 11, 13],)
        new_data = sort_item(data)
        new_data = {key.split('(')[0]: value for key, value in data.items()}
        
        os.makedirs(f'./data/{date}/items', exist_ok=True)

        with pd.ExcelWriter(f'./data/{date}/items/new_{file_name}', engine='xlsxwriter') as writer:
            for i in new_data.keys():
                tmp = new_data[i]
                tmp.to_excel(writer, sheet_name=f'{i}', index=True)
            print("new_data 정리하기 완료")
    else:
        print('##################### 데이터 읽어오기 선택')
        file_path = f'./data/{date}/items/new_{file_name}'
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        new_data = {}
        for sheet in sheet_names:
            new_data[sheet] = pd.read_excel(file_path, sheet_name=sheet)
        print("new_data 읽어오기 완료")
    return new_data

class Risk_Parity:
    """ Risk Parity Strategy

    Parameters
    ----------------------------
    price : 수정주가 데이터
    param : 공분산 주기
    """

    def __init__(self, price, param=52):

        # 연율화 패러미터
        self.param = param

        # 일별 수익률
        self.rets = price.pct_change().dropna()

        # 공분산행렬
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])

        # 거래비용
        self.cost = 0.00015

    # RP
    def rp(self, cov):
        noa = cov.shape[0]
        init_guess = np.repeat(1/noa, noa)
        bounds = ((0.0, 1.0), ) * noa
        target_risk = np.repeat(1/noa, noa)
        
        weights_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}
        
        def msd_risk(weights, target_risk, cov):
            
            port_var = weights.T @ cov @ weights
            marginal_contribs = cov @ weights
            
            risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
            
            w_contribs = risk_contribs
            return ((w_contribs - target_risk)**2).sum()
        
        weights = minimize(msd_risk, 
                            init_guess,
                            args=(target_risk, cov), 
                            method='SLSQP',
                            constraints=(weights_sum_to_1,),
                            bounds=bounds)
        return weights.x
    
    # 거래비용 함수
    def transaction_cost(self, weights_df, rets_df, cost=0.00015):
        # 이전 기의 투자 가중치
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])) \
        .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1), axis=0)

        # 거래비용 데이터프레임
        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)

        return cost_df

    # 백테스팅 실행 함수
    def run(self, cost):
        # 빈 딕셔너리
        backtest_dict = {}
        
        # 일별 수익률 데이터프레임
        rets = self.rets
        for i, index in enumerate(rets.index[engine.param-1:]):
            backtest_dict[index] = self.rp(self.cov[i])
        
        # 횡적 가중치 데이터프레임
        port_weights = pd.DataFrame(list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)
        port_weights.fillna(0, inplace=True)

        # 자산 수익률
        pf_rets = port_weights.shift(1) * rets.iloc[self.param-1:,:]

        # 포트폴리오 수익률
        port_rets = pf_rets.sum(axis=1)
    
        # 거래비용 데이터프레임
        cost = self.transaction_cost(port_weights, rets)

        # 최종 포트폴리오 자산별 수익률
        port_asset_rets = port_weights.shift() * rets - cost

        # 최종 포트폴리오 수익률 
        port_rets = port_asset_rets.sum(axis=1)
        port_rets.index = pd.to_datetime(port_rets.index).strftime("%Y-%m-%d")

        return port_weights, port_asset_rets, port_rets
    
    # 성과분석 수행 함수
    def performance_analytics(self, port_weights, port_asset_rets, port_rets, qs_report=False):
        
        # 자산별 투자 가중치
        plt.figure(figsize=(12, 7))
        port_weights['Cash'] = 1 - port_weights.sum(axis=1)
        plt.stackplot(port_weights.index, port_weights.T, labels=port_weights.columns)
        plt.title('Portfolio Weights')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.legend(loc='upper left')
        plt.show()

        # 자산별 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_asset_rets).cumprod() - 1)
        plt.title('Underlying Asset Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend(port_asset_rets.columns, loc='upper left')
        plt.show()

        # 포트폴리오 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_rets).cumprod() - 1)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.show()

        # QuantStats 성과분석 리포트 작성
        if qs_report == True:
            port_rets.index = pd.to_datetime(port_rets.index)
            qs.reports.html(port_rets, output='./strategy/file-name.html')

    # 비중

if __name__ == "__main__":
    file_name = "DBGAPS2024_pricevolume2.xlsx"
    # date = pd.Timestamp.now().strftime("%Y%m%d")
    date = "20240518"
    # 데이터 불러오기
    data = get_excel_data(date, file_name)

    # price 데이터
    price = data['수정주가']
    price.index = pd.to_datetime(price.index)
    gdu.data = price

    #TODO : 리밸런싱
    
    # 백테스팅 실행
    engine = Risk_Parity(price,param=4)
    res = engine.run(cost=0.00015)

    port_weights = res[0]
    port_asset_rets = res[1]
    port_rets = res[2]

    # 백테스팅 결과 시각화
    engine.performance_analytics(port_weights, port_asset_rets, port_rets, qs_report=True)

    port_weights