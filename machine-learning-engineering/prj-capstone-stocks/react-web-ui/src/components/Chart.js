import React, { useEffect } from "react";
import * as d3 from "d3";

const toOptNum = num => num === null || num === undefined ? num : +num

function Chart({ seriesData }) {
  // set the dimensions and margins of the graph
  const margin = { top: 20, right: 20, bottom: 50, left: 70 };
  const width = 960 - margin.left - margin.right;
  const height = 500 - margin.top - margin.bottom;

  const parseTime = d3.timeParse("%Y-%m-%d");
  // console.log('Raw data: ', JSON.parse(JSON.stringify(data)))
  const data = seriesData.map((sd) => ({
    date: parseTime(sd.date),
    value: toOptNum(sd.value),
    value2: toOptNum(sd.value2)
  }));
  // console.log(data)

  useEffect(() => {

    // append the svg object to the body of the page
    const g = d3.select("svg g")

    // add X axis and Y axis
    const x = d3.scaleTime().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);

    x.domain(d3.extent(data, (d) => { return d.date; }));
    // y.domain([0, d3.max(data, (d) => { return d.value; })]);
    // y.domain(d3.extent(data, d => d.value));
    y.domain([
        d3.min(data, d => Math.min(d.value, d.value2)),
        d3.max(data, d => Math.max(d.value, d.value2))
    ]);

    g.append("g")
      .attr("transform", `translate(0, ${height})`)
      .call(d3.axisBottom(x));

    g.append("g")
      .call(d3.axisLeft(y));

    // add the Line
    const valueLine = d3.line()
    .x((d) => { return x(d.date); })
    .y((d) => { return y(d.value); });

    g.append("path")
      .data([data])
      .attr("class", "line")
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("d", valueLine);

    const valueLine2 = d3.line()
    .x((d) => { return x(d.date); })
    .y((d) => { return y(d.value2); });

    g.append("path")
      .data([data])
      .attr("class", "line")
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 1.5)
      .attr("d", valueLine2);
  }, []);

  return (
    <svg
        width={width + margin.left + margin.right}
        height={height + margin.top + margin.bottom}
    >
      <g transform={`translate(${margin.left}, ${margin.top})`} />
    </svg>
  );
}

export default Chart