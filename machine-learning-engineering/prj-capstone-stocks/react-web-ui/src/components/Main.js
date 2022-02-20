import './Main.css';
import Form from './Form';

function Main() {
  return (
      <div className="bg-light">
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
        <div className="">
          <Form />
        </div>
        <div className="py-5">
          <div className="container">
            <div className="row">
              <div className="col-md-12 order-md-2">
                <h4 className="d-flex justify-content-between mb-3">
                    <span className="text-muted"><b>Results</b></span>
                </h4>
                <div className="card p-2 my-4">
                  <span>chart</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="py-5 text-muted text-center">
          <div className="container">
            <div className="row">
              <div className="col-md-12 my-4">
                <p className="mb-1">© 2022 Edgar Villegas</p>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
}

export default Main;
