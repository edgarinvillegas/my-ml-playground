import { useEffect, useState } from 'react';
import './Main.css';
import Form from './Form';
import Chart from './Chart'
import csvtojson from 'csvtojson'
import { useFetch } from "react-async"
import * as d3 from "d3";

const useStockFetch = (api, method='GET') => {
    const baseApiEndpoint = 'https://6pjyfcf6t1.execute-api.us-west-2.amazonaws.com/v1';
    return useFetch(`${baseApiEndpoint}/${api}`, {
        headers: { accept: "application/json" },
        method
    })
};
/*
const APIEndPoint = 'https://6pjyfcf6t1.execute-api.us-west-2.amazonaws.com/v1/read-results?trainingJobName=AAPL-f10-b30-2022-02-20-04-11-13-688429'
const APIResult = () => {
  const { data, error } = useFetch(APIEndPoint, {
    headers: { accept: "application/json" },
  })
  if (error) return <p>{error.message}</p>
  if (data) return <p>{JSON.stringify(data)}</p>
  return null
}
*/
const apiFetch$ = async (api, method='GET') => {
    const baseApiEndpoint = 'https://6pjyfcf6t1.execute-api.us-west-2.amazonaws.com/v1';
    const rawResponse = await fetch(`${baseApiEndpoint}/${api}`, {
        method,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        //body: JSON.stringify({a: 1, b: 'Textual content'})
    });
    return await rawResponse.json();
}

function Header() {
    return (
        <div className="py-5">
          <div className="container">
            <div className="row">
              <div className="text-center col-md-7 mx-auto">
                <i className="fa d-block fa-bullseye fa-5x mb-4 text-info" />
                <h2><b>Predict stock prices with AutoGluon</b></h2>
                <p className="lead">Capstone Project to predict and backtest stock prices (data comes live from Yahoo Finance)</p>
              </div>
            </div>
          </div>
        </div>
    );
}
function Footer() {
    return (
        <div className="py-5 text-muted text-center">
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

function Main() {
  const [data, setData] = useState(null)

  async function loadData() {
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
  }
  /*
  (async () => {
  const rawResponse = await fetch('https://httpbin.org/post', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({a: 1, b: 'Textual content'})
  });
  const content = await rawResponse.json();

  console.log(content);
})();
  * */

  const submitHandler = async ({ ticker, forecastMonths, lookbackMonths }) => {
      console.log(ticker, forecastMonths, lookbackMonths)
      //const obj = await csv().fromString(csvStr)
      const getDataResponse = await apiFetch$('get-data?ticker=GLD&forecastMonths=2&lookbackMonths=6&skipUpload=1', 'POST');
      console.log(getDataResponse)
  }

  useEffect(loadData, [])
  return (
      <div className="bg-light">
        <Header />
        <div className="">
          <Form onSubmit={submitHandler} />
        </div>
        <div className="py-5">
          <div className="container">
            <div className="row">
              <div className="col-md-12 order-md-2">
                <Results data={data} />
              </div>
            </div>
          </div>
        </div>
        <Footer />
      </div>
  );
}

function Results({ data }) {
    return (
        <>
            <h4 className="d-flex justify-content-between mb-3">
                    <span className="text-muted"><b>Results</b></span>
            </h4>
            <div className="card p-2 my-4">
                {!!data && <Chart key={data} data={data}/>}
            </div>
        </>
    )
}

export default Main;
