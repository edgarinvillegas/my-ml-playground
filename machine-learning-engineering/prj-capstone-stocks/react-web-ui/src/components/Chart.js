import React, { useEffect } from "react";
import * as d3 from "d3";

const toOptNum = num => num === null || num === undefined ? num : +num
const withValues = (arr) => arr.filter(x => x !== null && x !== undefined)

function Chart({ seriesData, ticker }) {
  // set the dimensions and margins of the graph
  const margin = { top: 20, right: 20, bottom: 50, left: 70 };
  // const width = 960 - margin.left - margin.right;

  const width = document.querySelector('.btn').clientWidth - margin.left - margin.right
  const height = 500 - margin.top - margin.bottom;
  // const [width, height] = [600, 400]

  const parseTime = d3.timeParse("%Y-%m-%d");
  // console.log('Raw data: ', JSON.parse(JSON.stringify(data)))
  const data = seriesData.map((sd) => ({
    date: parseTime(sd.date),
    value: toOptNum(sd.value),
    value2: toOptNum(sd.value2),
    value3: toOptNum(sd.value3)
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
    // y.domain([
    //     d3.min(data, d => Math.min(d.value, d.value2)),
    //     d3.max(data, d => Math.max(d.value, d.value2))
    // ]);

    y.domain(d3.extent(
        withValues([
            ...data.map(d => d.value),
            ...data.map(d => d.value2),
            ...data.map(d => d.value3)
        ]),
        x => x)
    );
    // Chart title
    g.append("text").attr("x", width / 2).attr("y", 0)
        .text(`${ticker} prices`)
        .style("font-size", "16px")
        .style("font-weight", 'bold')
        .attr("alignment-baseline","middle")

    g.append("g")
      .attr("transform", `translate(0, ${height})`)
      .call(d3.axisBottom(x));
    // X axis label
    g.append("text").attr("x", width / 2).attr("y", height + 35)
        .text("Date").style("font-size", "12px")
        .attr("alignment-baseline","middle")

    g.append("g")
      .call(d3.axisLeft(y));
    // Y axis label
    g.append("text").attr("x", -30).attr("y", -10)
        .text("Adj Close").style("font-size", "12px")
        .attr("alignment-baseline","middle")

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

    const valueLine3 = d3.line()
    .x((d) => { return x(d.date); })
    .y((d) => { return y(d.value3); });

    g.append("path")
      .data([data])
      .attr("class", "line")
      .attr("fill", "none")
      .attr("stroke", "green")
      .attr("stroke-width", 1.5)
      .attr("d", valueLine3);

    let legendX = -50 // 0
    let legendY = height + 40 //-10
    g.append("circle").attr("cx", legendX).attr("cy", legendY).attr("r", 6)
        .style("fill", "steelblue")
    g.append("text").attr("x", legendX + 8).attr("y", legendY)
        .text("Actual").style("font-size", "15px")
        .attr("alignment-baseline","middle")
    legendX += 60
    g.append("circle").attr("cx", legendX).attr("cy", legendY).attr("r", 6)
        .style("fill", "red")
    g.append("text").attr("x", legendX + 8).attr("y", legendY)
        .text("Predicted").style("font-size", "15px")
        .attr("alignment-baseline","middle")
    legendX += 80
    g.append("circle").attr("cx", legendX).attr("cy", legendY).attr("r", 6)
        .style("fill", "green")
    g.append("text").attr("x", legendX + 8).attr("y", legendY)
        .text("Benchmark").style("font-size", "15px")
        .attr("alignment-baseline","middle")

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