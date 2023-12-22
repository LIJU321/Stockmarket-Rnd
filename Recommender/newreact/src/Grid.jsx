import React from "react";
import { styled } from "@mui/material/styles";
import Grid from "@mui/material/Unstable_Grid2";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
// import Model from "./Model";

export default function GRID(props) {
  const Item = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
    ...theme.typography.body2,
    padding: theme.spacing(1),
    textAlign: "center",
    color: theme.palette.text.secondary,
  }));


  return (
    <div>
      <Box sx={{ width: "100%" }}>
        <Grid container rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }}>
          <Grid xs={6}>
            <Item>{props.Item}</Item>
          </Grid>
          <Grid xs={6}>
            <Item>{props.Item2}</Item>
          </Grid>
          <Grid xs={6}>
            <Item>{props.Item3}</Item>
          </Grid>
          <Grid xs={6}>
            <Item>{props.Item4}</Item>
          </Grid>
          <Grid xs={6}>
            <Item>{props.Item5}</Item>
          </Grid>
          <Grid xs={6}>
            <Item>{props.Item6}</Item>
          </Grid>
        </Grid>
      </Box>
    </div>
  );
}
