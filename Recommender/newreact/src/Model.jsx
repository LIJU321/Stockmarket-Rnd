import React from "react";
import { Button, TextField, Typography } from "@material-ui/core";
import { Form } from "react-bootstrap";
import { useState } from "react";
// import { useEffect } from "react";
import axios from "axios";

export default function Model(props) {
  const [Price, setPrice] = useState("");
  const [Data, setData] = useState("");

  // let myArray = Array.from(Data);
  // console.log(myArray);
  // console.log(Data)
  // console.log(Array.isArray(Data))
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
  // }, [Price]);
  // console.log(Data);

  // ///////////////////////////////

  const submit = (e) => {
    e.preventDefault();

    // var p = parseInt(Price)   //Parsint convert string into An Integer
    // console.log(p + 1)

    //////////////////////////////////////////////////////////////
    // axios
    //   .get(`http://127.0.0.1:8000/Cate/22LgEjLjvhCrzaQcT/${Price}/`)
    //   .then((response) => {
    //     console.log(response); // Access the data directly
    //     setData(response.data);
    //     // console.log(Data);
    //   })
    //   .catch((error) => {
    //     console.error("Error fetching data:", error);
    //   });

    //// STCOK PRICE ///
    // axios
    // .get(`http://127.0.0.1:8000/UI2/${Price}/`)
    // .then((response) => {
    //   console.log(response); // Access the data directly
    //   setData(response.data.Close_price);

    //   // console.log(Data);
    // })
    // .catch((error) => {
    //   console.error("Error fetching data:", error);
    // });

    axios
      .get(`http://127.0.0.1:8000/${props.alg}/${props.textfieldvalue}/`)
      .then((response) => {
        // console.log(response); // Access the data directly
        setData(response.data.Close_price);

        // console.log(Data);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
    
  };

  console.log(Price);
  return (
    <div id="maindiv">
      <div id="TodoimageDiv">
        <Typography id="Todoimage">{props.name}</Typography>
        <Form id="forms" onSubmit={submit}>
          <TextField
            name="value"
            value={props.textfieldvalue}
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

          <Button id="Loginbttn" type="submit" onClick={submit}>
            PREDICT
          </Button>
        </Form>
        <div id="LabelDiv">
          <TextField id="Label" type="text" value={Data} aria-readonly />
          {/* <input id="Label" type="text" value={Data} readOnly /> */}
          <label id="Label2">{props.pos}</label>
        </div>
        {/* <TextField id="Label" type="text"  value={Data} aria-readonly/>
        <label id="Label2">Close</label> */}
      </div>
    </div>
  );
}
