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
   * https://www.gorillasun.de/blog/an-algorithm-for-shapeBs-with-rounded-corners/
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



function mergeHulls(shapes) {
  let toBeMerged = [];
  for (var a = shapes.length - 1; a >= 0; a--) {
    for (var b = shapes.length - 1; b >= 0; b--) {
      if (a > b) {
        shapeA = getHull(shapes[a],40)
        shapeB = getHull(shapes[b],40)
        if (doesIntersect(shapeA,shapeB)) {
          toBeMerged = [a,b]
          continue;
        } else if (isInside(shapeA,shapeB)) {
          toBeMerged = [a,b]
          continue;
        } else if (isInside(shapeB,shapeA)) {
           toBeMerged = [a,b]
          continue;         
        }
      }
    }
  }
  if (toBeMerged.length > 0) {
    let newShape = shapes[toBeMerged[0]].concat(shapes[toBeMerged[1]])
    shapes[toBeMerged[0]] = newShape
    delete shapes[toBeMerged[1]]
    shapes = shapes.filter(n => n)
    return mergeHulls(shapes)
  } else {
    return shapes
  }
}

function doesIntersect(shapeA, shapeB) {
  // Check if any line segment of shapeA intersects with any line segment of shapeB
  for (let i = 0; i < shapeA.length; i++) {
    let j = (i + 1) % shapeA.length;
    for (let k = 0; k < shapeB.length; k++) {
      let l = (k + 1) % shapeB.length;
      if (doSegmentsIntersect(shapeA[i], shapeA[j], shapeB[k], shapeB[l])) {
        return true;
      }
    }
  }
  return false;
}

function doSegmentsIntersect(p1, p2, p3, p4) {
  // Check if line segments p1, p2 and p3, p4 intersect
  let d1 = direction(p3, p4, p1);
  let d2 = direction(p3, p4, p2);
  let d3 = direction(p1, p2, p3);
  let d4 = direction(p1, p2, p4);
  return (d1 != d2 && d3 != d4 && d1 != 0 && d2 != 0 && d3 != 0 && d4 != 0);
}

function direction(p1, p2, p3) {
  // Calculate the direction of the cross product (p3 - p1) x (p2 - p1)
  let val = (p2.y - p1.y) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.y - p2.y);
  if (val == 0) {
    return 0; // Collinear points
  } else if (val < 0) {
    return 1; // Clockwise direction
  } else {
    return 2; // Counterclockwise direction
  }
}

function isInside(shapeA, shapeB) {
  let count = 0;
  for (let i = 0; i < shapeA.length; i++) {
    let isInside = false;
    let testPoint = shapeA[i];
    for (let j = 0, k = shapeB.length - 1; j < shapeB.length; k = j++) {
      let pj = shapeB[j];
      let pk = shapeB[k];
      if (
        ((pj.y > testPoint.y) != (pk.y > testPoint.y)) &&
        (testPoint.x < ((pk.x - pj.x) * (testPoint.y - pj.y)) / (pk.y - pj.y) + pj.x)
      ) {
        isInside = !isInside;
      }
    }
    if (isInside) {
      count++;
    }
  }
  return count === shapeA.length;
}