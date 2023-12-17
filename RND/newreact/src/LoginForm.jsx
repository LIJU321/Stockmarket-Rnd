import { Typography } from "@material-ui/core";
import React from "react";
// import { Form } from "react-bootstrap";
import { useState } from "react";
// import { useEffect } from "react";

import { TextField } from "@material-ui/core";
import Model from "./Model";
import { Fragment } from "react";

export const LoginForm = () => {
  const [Price, setPrice] = useState("");

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

  // const submit = (e) => {
  //   e.preventDefault();

  //   // var p = parseInt(Price)   //Parsint convert string into An Integer
  //   // console.log(p + 1)

  //   //////////////////////////////////////////////////////////////
  //   // axios
  //   //   .get(`http://127.0.0.1:8000/Cate/22LgEjLjvhCrzaQcT/${Price}/`)
  //   //   .then((response) => {
  //   //     console.log(response); // Access the data directly
  //   //     setData(response.data);
  //   //     // console.log(Data);
  //   //   })
  //   //   .catch((error) => {
  //   //     console.error("Error fetching data:", error);
  //   //   });

  //   //// STCOK PRICE ///

  // };

  return (
    <Fragment>
      <div className="LoginformMainDiv">
        <div className="login-form">
          <div id="Tittle1">
            <Typography id="Tittle">Stock Prediction System</Typography>
          </div>
          <TextField
            name="value"
            value={Price}
            onChange={(event) => setPrice(event.target.value)}
            id="price"
            variant="outlined"
            placeholder="Enter Open Price"
            InputProps={{
              style: {
                maxWidth: "200px",
                maxHeight: "74px",
                width: "100%",
                height: "100%",
                marginLeft: "10px",
                backgroundColor: "white",
              },
            }}
          />
          <Model
            name={"Linear Regression"}
            alg={"Regression"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"KNN neighbours"}
            alg={"KNN"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"Neural Network"}
            alg={"Neural-Network"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"SVM"}
            alg={"SVM"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"RNN"}
            alg={"RNN"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"Dicision_Tree"}
            alg={"Dicision_Tree"}
            pos={"Close"}
            textfieldvalue={Price}
          />
          <Model
            name={"Dicision_Tree_on_High"}
            alg={"Dicision_Tree_on_High"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"HIgh_svmclassifier"}
            alg={"HIgh_svmclassifier"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"HIgh_price_on_Nerual_Network"}
            alg={"HIgh_price_on_Nerual_Network"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"HIgh_on_regression"}
            alg={"HIgh_on_regression"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"HIgh_price_on_Feed_forward_Neuaral_Network"}
            alg={"HIgh_price_on_Feed_forward_Neuaral_Network"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"HIgh_price_on_knn"}
            alg={"HIgh_price_on_knn"}
            pos={"High"}
            textfieldvalue={Price}
          />
          <Model
            name={"Low_on_regression"}
            alg={"Low_on_regression"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Model
            name={"Low_svmclassifier"}
            alg={"Low_svmclassifier"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Model
            name={"Low_price_on_knn"}
            alg={"Low_price_on_knn"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Model
            name={"Low_price_on_Nerual_Network"}
            alg={"Low_price_on_Nerual_Network"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Model
            name={"Low_on_Feed_forward_Neuaral_Network"}
            alg={"Low_on_Feed_forward_Neuaral_Network"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Model
            name={"Dicision_Tree_on_Low"}
            alg={"Dicision_Tree_on_Low"}
            pos={"Low"}
            textfieldvalue={Price}
          />
          <Typography className="Loginlabel">Adani Ports </Typography>
        </div>
      </div>
    </Fragment>
  );
};
export default LoginForm;
