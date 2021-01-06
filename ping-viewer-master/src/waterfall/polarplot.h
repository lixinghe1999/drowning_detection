#pragma once

#include <QImage>
#include <QQuickPaintedItem>
#include <QTimer>

#include "logger.h"
#include "ringvector.h"
#include "waterfall.h"
#include "waterfallgradient.h"

Q_DECLARE_LOGGING_CATEGORY(waterfall)

/**
 * @brief Polar widget
 *
 */
class PolarPlot : public Waterfall {
    Q_OBJECT

public:
    /**
     * @brief Construct a new PolarPlot object
     *
     * @param parent
     */
    PolarPlot(QQuickItem* parent = nullptr);

    /**
     * @brief This is used by the qml paint event
     *  This paint the a polar waterfall in qml
     *
     * @param painter
     */
    void paint(QPainter* painter) final override;

    /**
     * @brief Set the polar Image
     *
     * @param image
     */
    void setImage(const QImage& image);

    /**
     * @brief Draw a list of points in the waterfall
     *
     * @param points
     * @param angle
     * @param initPoint
     * @param length
     * @param angleGrad
     * @param sectorSize
     */
    Q_INVOKABLE void draw(
        const QVector<double>& points, float angle, float initPoint, float length, float angleGrad, float sectorSize);

    /**
     * @brief Clear waterfall and restart all parameters
     *
     */
    Q_INVOKABLE void clear() final override;

    /**
     * @brief Return waterfall image
     *
     * @return QImage
     */
    QImage image() { return _image; }
    Q_PROPERTY(QImage image READ image WRITE setImage NOTIFY imageChanged)

    /**
     * @brief Get angle from mouse position
     *
     * @return float
     */
    float mouseSampleAngle() { return _mouseSampleAngle; }
    Q_PROPERTY(float mouseSampleAngle READ mouseSampleAngle NOTIFY mouseSampleAngleChanged)

    /**
     * @brief Get distance from mouse position
     *
     * @return float
     */
    float mouseSampleDistance() { return _mouseSampleDistance; }
    Q_PROPERTY(float mouseSampleDistance READ mouseSampleDistance NOTIFY mouseSampleDistanceChanged)

    /**
     * @brief Max distance in waterfall
     *
     * @return float
     */
    float maxDistance() { return _maxDistance; }
    Q_PROPERTY(float maxDistance READ maxDistance NOTIFY maxDistanceChanged)

signals:
    void imageChanged();
    void maxDistanceChanged();
    void mouseSampleAngleChanged();
    void mouseSampleDistanceChanged();
    void sectorSizeDegreesChanged();

private:
    Q_DISABLE_COPY(PolarPlot)

    /**
     * @brief Update mouse column information
     *
     */
    void updateMouseColumnData();

    QVector<float> _distances;
    QImage _image;
    float _maxDistance;
    float _mouseSampleAngle;
    float _mouseSampleDistance;
    QPainter* _painter;
    float _sectorSizeDegrees;
    static uint16_t _angularResolution;
    QTimer _updateTimer;
};
