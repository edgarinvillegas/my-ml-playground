import { useState } from 'react'

const tickers = {
  'AAL': 'American Airlines Group Inc.',
  'AAPL': 'Apple Inc.',
  'AMC': 'AMC Entertainment Holdings, Inc.',
  'AMD': 'Advanced Micro Devices, Inc.',
  'AZN': 'AstraZeneca PLC',
  'BAC': 'Bank of America Corporation',
  'BBD': 'Banco Bradesco S.A.',
  'CCL': 'Carnival Corporation & plc',
  'CSCO': 'Cisco Systems, Inc.',
  'DKNG': 'DraftKings Inc.',
  'F': 'Ford Motor Company',
  'FB': 'Meta Platforms, Inc.',
  'GOLD': 'Barrick Gold Corporation',
  'INTC': 'Intel Corporation',
  'ITUB': 'Itaú Unibanco Holding S.A.',
  'MSFT': 'Microsoft Corporation',
  'NIO': 'NIO Inc.',
  'NLY': 'Annaly Capital Management, Inc.',
  'NU': 'Nu Holdings Ltd.',
  'NVDA': 'NVIDIA Corporation',
  'PLTR': 'Palantir Technologies Inc.',
  'RBLX': 'Roblox Corporation',
  'ROKU': 'Roku, Inc.',
  'SOFI': 'SoFi Technologies, Inc.',
  'T': 'AT&T Inc.',
};

function Form({ onSubmit }) {
    const [ticker, setTicker] = useState('AAL')
    const [ forecastMonths, setForecastMonths ] = useState(1)
    const [ lookbackMonths, setLookbackMonths ] = useState(3)

    const submitHandler = () => {
      onSubmit({ ticker, forecastMonths, lookbackMonths })
    }
    const tickerChangeHandler = event => setTicker(event.target.value)
    const forecastChangeHandler = event => setForecastMonths(event.target.value)
    const lookbackChangeHandler = event => setLookbackMonths(event.target.value)

    return (
        <form className="needs-validation" noValidate="">
            <div className="container">
              <div className="row">
                <div className="col-md-6">
                  <div className="mb-3">
                    <label htmlFor="ticker">Ticker</label>
                    <select className="custom-select d-block w-100" id="ticker" required={true} value={ticker} onChange={tickerChangeHandler}>
                      {Object.entries(tickers).map(([key, text]) => (
                          <option key={key} value={key}>{key} - {text}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="col-md-3">
                  <div className="mb-3"><label htmlFor="forecast">Forecast <span
                      className="text-muted">(Months)</span></label>
                    <input type="number" step="1" value={forecastMonths} min="1" max="12" className="form-control" id="forecast"
                           placeholder="Months to predict" onChange={forecastChangeHandler} />
                  </div>
                </div>
                <div className="col-md-3">
                  <div className="mb-3"><label htmlFor="lookback">Lookback <span
                      className="text-muted">(Months)</span></label>
                    <input type="number" step="1" value={lookbackMonths} min="1" max="120" className="form-control" id="lookback"
                           placeholder="Months to look back" onChange={lookbackChangeHandler} />
                  </div>
                </div>
              </div>
              <div className="row">
                <div className="col-md-12">
                  <button className="btn btn-primary btn-lg btn-block" type="button" onClick={submitHandler}>Backtest forecast</button>
                </div>
              </div>
            </div>
          </form>
    )
}

export default Form;