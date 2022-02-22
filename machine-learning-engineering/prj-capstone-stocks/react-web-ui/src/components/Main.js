import { useState } from 'react';
import './Main.css';
import Form from './Form';
import Chart from './Chart'
import csvtojson from 'csvtojson'
import { apiFetch$ } from '../utils'

function Main() {
  const [ticker, setTicker] = useState('')
  const [testData, setTestData] = useState(null)
  const [trainData, setTrainData] = useState(null)
  const [trainingJobName, setTrainingJobName] = useState('')
  const [predictions, setPredictions] = useState(null)
  const [inferenceResults, setInferenceResults] = useState(null)

  /*async function loadData() {
      // read data from csv and format variables
    const tmpData = await d3.csv('https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/3_TwoNumOrdered_comma.csv')
    setData(tmpData)
    setTimeout(() => {
        console.log('Setting data again...')
        // setData(JSON.parse(JSON.stringify(tmpData)))
        setData([...tmpData, {
            date: '2022-02-20',
            value: '9999.99'
        }])
    }, 10000)
  }*/

  const reset = () => {
      setTicker('')
      setTestData(null)
      setTrainData(null)
      setTrainingJobName('')
      setPredictions(null)
      setInferenceResults(null)
  }

  const submitHandler = async ({ ticker, forecastMonths, lookbackMonths }) => {
      console.log(ticker, forecastMonths, lookbackMonths)
      reset()
      setTicker(ticker)
      //const obj = await csv().fromString(csvStr)
      // const getDataResponse = await apiFetch$('get-data?ticker=GLD&forecastMonths=2&lookbackMonths=6&skipUpload=0', 'POST');
      const getDataResponse = await apiFetch$(`get-data?ticker=${ticker}&forecastMonths=${forecastMonths}&lookbackMonths=${lookbackMonths}&skipUpload=0`, 'POST');
      setTrainingJobName(getDataResponse.trainingJobName);
      // console.log(getDataResponse)
      const testData = await csvtojson().fromString(getDataResponse.testCsv)
      setTestData(testData)
      const trainData = await csvtojson().fromString(getDataResponse.trainCsv)
      setTrainData(trainData)
      startPollingResults(getDataResponse.trainingJobName)
  }

  const MOCK_submitHandler = async ({ ticker, forecastMonths, lookbackMonths }) => {
      console.log(ticker, forecastMonths, lookbackMonths)
      reset()
      setTicker(ticker)
      const getDataResponse = await apiFetch$('get-data?ticker=GLD&forecastMonths=2&lookbackMonths=6&skipUpload=1', 'POST');
      setTrainingJobName(getDataResponse.trainingJobName);
      // console.log(getDataResponse)
      const testData = await csvtojson().fromString(getDataResponse.testCsv)
      setTestData(testData)
      const trainData = await csvtojson().fromString(getDataResponse.trainCsv)
      setTrainData(trainData)
      MOCK_startPollingResults(getDataResponse.trainingJobName)
  }

  const MOCK_startPollingResults = async (trainingJobName) => {
      await delay$(5000)
      const readResultsResponse = await apiFetch$('read-results?trainingJobName=GLD-f2-b6-2022-02-20-23-48-23-279316');
      onResultsReady(readResultsResponse)
  }


  const delay$ = (delay) => {
      return new Promise((res, rej) => setTimeout(res, delay))
  };

  const startPollingResults = async (trainingJobName) => {
      let readResultsResponse = null
      let i = 1
      do {
          console.log(`Attempt ${i++} of polling...`)
          try{
              readResultsResponse = await apiFetch$(`read-results?trainingJobName=${trainingJobName}`);
              console.log('readResultsResponse: ', readResultsResponse)
          } catch (exc) {
              console.log('Error on read-results. Keep trying... ', exc)
          }
          await delay$(5000)
      } while(readResultsResponse === null || readResultsResponse.error);
      onResultsReady(readResultsResponse)
  }

  const onResultsReady = async (inferenceResults) => {
      const predictions = await csvtojson().fromString(inferenceResults.predictions)
      setPredictions(predictions)
      setInferenceResults(inferenceResults)
      // console.log('predictions', predictions)
  };

  // Quick fix for data mismatch
  if(predictions && testData && predictions.length !== testData.length) predictions.length = testData.length = Math.min(predictions.length, testData.length)

  /*const movingAverage = (arr, i, n) => {
      if(i < n) return null;
      const nArr = arr.slice(i - n, i);
      return nArr.reduce((a, b) => a + b, 0) / nArr.length;
  }
  */
  const movingAverage = (arr, i, n) => {
      const nArr = arr.slice(Math.max(i - n, 0), i + 1)
      return nArr.reduce((a, b) => a + b, 0) / nArr.length || arr[0];
  }

  const daysBackForMovingAverage = 7
  const mergedPredictions = testData ? testData.map((t, i) => ({
      date: t.Date,
      value: t.Adj_Close,
      value2: predictions && predictions[i] ? predictions[i].Predicted : null,
      // value3: +t.Adj_Close * (0.995 + Math.random()*0.01)
      value3: movingAverage(testData.map(t => +t.Adj_Close), i, daysBackForMovingAverage)
  })): null

  // console.log('mergedPredictions', mergedPredictions)
  console.log('value3s: ', mergedPredictions && mergedPredictions.map(x => x.value3))

  // useEffect(loadData, [])
  return (
      <div className="bg-light">
          <Header />
          <div className="">
              <Form onSubmit={submitHandler}/>
          </div>
          <div className="py-3">
              {trainingJobName && <Progress
                  testData={testData}
                  trainingJobName={trainingJobName}
                  predictions={predictions}
              />}
          </div>
          <div className="py-3">
              <div className="container">
                  <div className="row">
                      <div className="col-md-12 order-md-2">
                          {!!mergedPredictions && (
                              <Results
                                  seriesData={mergedPredictions}
                                  ticker={ticker}
                                  inferenceResults={inferenceResults}
                              />
                          )}
                      </div>
                  </div>
              </div>
          </div>
          <Footer/>
      </div>
  );
}

function Progress({ testData, trainingJobName, predictions }) {
    const isDataReady = !!testData && !!testData.length
    const isInferenceReady = !!predictions && !!predictions.length
    return (
        <div className="container">
            <div className="row">
                <div className="col-md-12">
                    <h4>Progress</h4>
                    <ul className="list-group list-group-flush" style={{listStyleType: 'none'}}>
                        {isDataReady && (
                            <li>
                                <i className="fa fa-check text-primary mr-2" style={{color: 'green'}}/>Data gathered
                            </li>
                        )}
                        <li>
                            {isInferenceReady ? (
                                <>
                                    <i className="fa fa-check text-primary mr-2"/>
                                    Trained.
                                </>
                            ): 'Training... '}
                             <span style={{ marginLeft: 10, fontStyle: 'italic'}}>(Job name: {trainingJobName})</span>
                        </li>
                        {isInferenceReady && (
                            <>
                                <li>
                                    <i className="fa fa-check text-primary mr-2"/>Inference
                                </li>
                                <li>
                                    Done
                                </li>
                            </>
                        )}
                    </ul>
                </div>
            </div>
        </div>
    );
}


function Results({ seriesData, ticker, inferenceResults }) {
    return (
        <>
            {/*<h4 className="d-flex justify-content-between mb-3">*/}
            {/*        <span className="text-muted"><b>Results</b></span>*/}
            {/*</h4>*/}
            <h2>Results</h2>
            <div className="card p-2 my-4">
                <Chart key={JSON.stringify(seriesData)} seriesData={seriesData} ticker={ticker}/>
            </div>
            {!!inferenceResults && (
                <>
                    <div className="card p-2 my-4">
                        <h4>Model performance</h4>
                        <pre>
                            {inferenceResults.model_performance}
                        </pre>
                    </div>
                    {/*<div className="card p-2 my-4">*/}
                    {/*    <h4>Models leaderboard</h4>*/}
                    {/*    <pre>*/}
                    {/*        {inferenceResults.leaderboard}*/}
                    {/*    </pre>*/}
                    {/*</div>*/}
                    {/*<div className="card p-2 my-4">*/}
                    {/*    <h4>Fit summary</h4>*/}
                    {/*    <pre>*/}
                    {/*        {inferenceResults.fit_summary}*/}
                    {/*    </pre>*/}
                    {/*</div>*/}
                </>
            )}
        </>
    )
}

function Header() {
    return (
        <div className="py-5">
          <div className="container">
            <div className="row">
              <div className="text-center col-md-7 mx-auto">
                <i className="fa d-block fa-bullseye fa-5x mb-4 text-info" />
                <h2><b>Predict stock prices with AutoGluon</b></h2>
                <p className="lead">Capstone Project to predict and backtest stock prices (with Yahoo Finance live data)</p>
              </div>
            </div>
          </div>
        </div>
    );
}
function Footer() {
    return (
        <div className="py-3 text-muted text-center">
          <div className="container">
            <div className="row">
              <div className="col-md-12 my-4">
                <p className="mb-1">© 2022 Edgar Villegas</p>
              </div>
            </div>
          </div>
        </div>
    );
}

export default Main;
