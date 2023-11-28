import { Typography } from "@material-ui/core";
import React  from "react";
// import { Form } from "react-bootstrap";
// import { useState } from "react";
// import { useEffect } from "react";
// import axios from "axios";
import Model from "./Model";
import { Fragment } from "react";

export const LoginForm = () => {
  // const [Price, setPrice] = useState("");

  // const [Data, setData] = useState("");
  // console.log(Data)

  // useEffect(() => {
  //   axios.get("https://jsonplaceholder.typicode.com/users")
  //     .then((response) => {
  //       console.log(response.data); // Access the data directly
  //       setData(response.data);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching data:", error);
  //     });
  // }, []);

  ////////////////////////////////

  // useEffect(() => {
  //   axios.get("http://127.0.0.1:8000/Cate/22LgEjLjvhCrzaQcT/50/")
  //     .then((response) => {
  //       console.log(response.data); // Access the data directly
  //       setData(response.data);
  //       // console.log(Data);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching data:", error);
  //     });
  // }, []);
  // console.log(Data);

  ////////////////////////////////

  // useEffect(() => {
  //   axios.get(`http://127.0.0.1:8000/Cate/22LgEjLjvhCrzaQcT/${Price}/`)
  //     .then((response) => {
  //       console.log(response); // Access the data directly
  //       setData(response.data);
  //       // console.log(Data);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching data:", error);
  //     });
  // },[Price]);
  // console.log(Data);

  // ///////////////////////////////

  //   const submit = (e) => {
  //     e.preventDefault();
  //     console.log("SEND")
  //     // var p = parseInt(Price)   //Parsint convert string into An Integer
  //     // console.log(p + 1)
    
  // //////////////////////////////////////////////////////////////
  //       axios.get(`http://127.0.0.1:8000/Cate/22LgEjLjvhCrzaQcT/${Price}/`)
  //     .then((response) => {
  //       console.log(response); // Access the data directly
  //       setData(response.data);
  //       // console.log(Data);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching data:", error);
  //     });

  // /////////////////////////////////////////////////////////////

     
  //   };


  return (
    <Fragment>
    <div className="LoginformMainDiv">
      <div className="login-form">
        <div id="Tittle1">
          <Typography id="Tittle">Stock Prediction System</Typography>
        </div>
        <Model name={"Linear Regression"}  alg = {"UI2"} />
        <Model  name={"KNN neighbours"}  alg = {"UI3"} />
        <Model  name={"DNN"}  alg = {"UI4"} />
        <Model  name={"Neural Network"}  alg = {"UI5"} />
        <Model  name={"SVM"}  alg = {"UI6"} />
        <Typography className="Loginlabel">Adani Ports </Typography>
      </div>
    </div>
    </Fragment>

    
  );
};
export default LoginForm;
