/*
 * pieces of code / methods borrowed from external sources
 */

function centerOfGravity(points) {
  let xs = points.reduce((acc,p) => acc + p.x, 0);
  let ys = points.reduce((acc,p) => acc + p.y, 0);
  let nb = points.length; 
  return {'x':xs/nb,'y':ys/nb};
}

const range = (start, end, length = end - start) => Array.from({ length }, (_, i) => start + i)

function drawRoundedShape(points,r,canvas,z) {

  /*
   * we use the rounded shape method from :
   * https://www.gorillasun.de/blog/an-algorithm-for-polygons-with-rounded-corners/
   * we just add re-scaling factor
   */

  let cg = centerOfGravity(points)
  canvas.translate(cg.x,cg.y)
  canvas.beginShape()
  for (let i = 0; i < points.length; i++) {
    const a = points[i]
    const b = points[(i+1) % points.length]
    const c = points[(i+2) % points.length]
    const ba = a.copy().sub(b).normalize()
    const bc = c.copy().sub(b).normalize()
    const normal = ba.copy().add(bc).normalize()
    const theta = ba.angleBetween(bc)
    const maxR = min(a.dist(b), c.dist(b))/2 * abs(sin(theta / 2))
    const cornerR = min(r, maxR)
    const distance = abs(cornerR / sin(theta / 2))
    if (!isNaN(distance)) {
      const c1 = b.copy().add(ba.copy().mult(distance))
      const c2 = b.copy().add(bc.copy().mult(distance))
      const bezierDist = 0.5523 // https://stackoverflow.com/a/27863181
      const p1 = c1.copy().sub(ba.copy().mult(2*cornerR*bezierDist))
      const p2 = c2.copy().sub(bc.copy().mult(2*cornerR*bezierDist))
      canvas.vertex(c1.x - cg.x, c1.y - cg.y)
      canvas.bezierVertex(
        p1.x - cg.x, p1.y - cg.y,
        p2.x - cg.x, p2.y - cg.y,
        c2.x - cg.x, c2.y - cg.y
      )
    }
  }
  canvas.endShape(CLOSE)
}  

  // draw an imperfectCircle from https://www.fxhash.xyz/article/drawing-imperfect-circles

function imperfectCircle(c,numVertices,radius,xo,yo) {

  let noise = .1;

  let vertices = [];
  for (let i = 0; i < numVertices; i++) {
    const rad = i * 2 * PI / numVertices;
    const x = radius * cos(rad) * random(1 - noise, 1 + noise);
    const y = radius * sin(rad) * random(1 - noise, 1 + noise);

    vertices.push({ x: xo + x, y: yo - y });
  }

  // Duplicate 3 vertices to close the loop and provide guides for the first and last points.
  for (let i = 0; i < 3; i++) {
    vertices.push(vertices[i]);
  }

  c.beginShape();
  for (let i = 0; i < vertices.length; i++) {
    c.curveVertex(vertices[i].x, vertices[i].y);
  }
  c.endShape();
}

function getHull(points,growth) {
  let original_hull = points
  if (points.length > 2) {
    original_hull = findConvexHull(points)
  }
  let extended_points = []
  original_hull.forEach((p) => {
    extended_points.push(p)
    extended_points.push(createVector(p.x + growth,p.y))
    extended_points.push(createVector(p.x - growth,p.y))
    extended_points.push(createVector(p.x,p.y + growth))
    extended_points.push(createVector(p.x,p.y - growth))
  })
  let extended_hull = findConvexHull(extended_points)
  return extended_hull
}


function findConvexHull(points) {

  /*
   * we use the gift wrapping algorithm from D. Shiffman :
   * https://editor.p5js.org/codingtrain/sketches/IVE9CxBOF
   */

  let hull = [];
  points.sort((a,b) => a.x - b.x);
  let leftMost = points[0]
  let curr = leftMost
  hull.push(curr)
  let next = points[1]
  let idx = 2;
  while (true) {
    let checking = points[idx]
    let a = p5.Vector.sub(next,curr)
    let b = p5.Vector.sub(checking,curr)
    let cross = a.cross(b)
    if (cross.z < 0) {
      next = checking
    }
    idx += 1
    if (idx == points.length) {
      if (next == leftMost){
        break          
      }
      idx = 0
      hull.push(next)
      curr = next
      next = leftMost
    }
  }
  return hull;
}  


function filterEdges(alpha) {
  
  /*
   * we use the disparity filter algorithme from :
   * Serrano, M. Á., Boguná, M., & Vespignani, A. (2009). Extracting the multiscale backbone 
   * of complex weighted networks. Proceedings of the national academy of sciences, 106(16), 6483-6488.
   */

  periods.forEach((period) => {
    (nodes.findRows(period, 'period_int')).forEach((node) => {
      let i = (node.get('id')).split('_')[0]
      let k = getDegree(i,period)
      let si = getStrength(i,period)
      let neighbours = getNeighbours(i,period)
      let nj = neighbours.map((e) => e.get('id'))
      let aj = neighbours.map((e) => {
        let pij = parseFloat(e.get('weight')) / si
        if ((1 - pij) == 0) {
          return 0
        } else {
          return Math.pow((1 - pij),(k - 1))
        }
      })
      let disparity = zip(nj,aj)
      let connected = false 
      Object.keys(disparity).forEach((j) => {
        if (disparity[j] <= alpha) {
          raw_sync_edges.set(j,'filtered',true)
        } 
        connected = raw_sync_edges.get(j,'filtered')
      })
      // we avoid disconnected networks
      if ((!connected)) {
        let tmp = 1;
        let best = 0
        Object.keys(disparity).forEach((j) => {
          if (disparity[j] < tmp) {
            tmp = disparity[j]
            best = j
          }
        })
        raw_sync_edges.set(best,'filtered',true)
      }
    })
  })
}