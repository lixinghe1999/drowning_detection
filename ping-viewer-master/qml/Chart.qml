import QtCharts 2.2
import QtQuick 2.15
import Util 1.0

Item {
    id: root

    property bool flip: false
    property real maxDepthToDraw: 0
    property real minDepthToDraw: 0

    function correctChartSize() {
        chart.height = width;
        // plotArea.x is a private margin in ChartView
        // it's not possible to set as 0
        chart.width = height + 2 * chart.plotArea.x;
    }

    function draw(points, depth, initPos) {
        chart.draw(points, depth, initPos);
        correctChartSize();
    }

    anchors.margins: 0
    onWidthChanged: correctChartSize()
    onHeightChanged: correctChartSize()

    ChartView {
        id: chart

        function draw(points, depth, initPost) {
            if (points) {
                // If there is no user configuration, we set the max and min automatically
                if (maxDepthToDraw == 0 && minDepthToDraw == 0) {
                    Util.update(serie, points, initPost, depth, initPost, depth, 1);
                    Util.update(serieInv, points, initPost, depth, initPost, depth, -1);
                } else {
                    Util.update(serie, points, initPost, depth, minDepthToDraw, maxDepthToDraw, 1);
                    Util.update(serieInv, points, initPost, depth, minDepthToDraw, maxDepthToDraw, -1);
                }
                return ;
            }
            serie.clear();
            serieInv.clear();
            for (var i = 0; i < 200; i++) {
                var pt = 255 * Math.random();
                serie.append(i, pt);
                serieInv.append(i, -pt);
            }
        }

        width: root.height
        height: root.width
        antialiasing: true
        legend.visible: false
        backgroundColor: "transparent"
        plotAreaColor: "transparent"
        backgroundRoundness: 0
        margins.top: 0
        margins.bottom: 0
        margins.left: 0
        margins.right: 0
        anchors.margins: 0
        Component.onCompleted: {
            for (var id in axes) {
                axes[id].visible = false;
            }
        }

        anchors {
            horizontalCenter: parent.horizontalCenter
            verticalCenter: parent.verticalCenter
        }

        ValueAxis {
            id: axisX

            //titleText: qsTr("Distance [m]")
            visible: false
            lineVisible: false
            gridVisible: false
            labelsAngle: -90
            labelsVisible: false
            labelFormat: "%.1f"
            tickCount: 3
            min: 0
            max: 2000
        }

        ValueAxis {
            id: axisY

            //titleText: qsTr("Intensity [dB]")
            visible: false
            lineVisible: false
            gridVisible: false
            labelsAngle: -90
            labelsVisible: false
            labelFormat: "%.1f"
            tickCount: 2
            min: -1
            max: 1
        }

        LineSeries {
            id: serie

            pointLabelsVisible: false
            pointsVisible: false
            axisX: axisX
            axisY: axisY
            color: "lime"
        }

        LineSeries {
            id: serieInv

            pointLabelsVisible: false
            pointsVisible: false
            axisX: axisX
            axisY: axisY
            color: "lime"
        }

    }

    transform: Rotation {
        origin.x: width / 2
        origin.y: height / 2
        angle: flip ? -90 : 90
    }

}
