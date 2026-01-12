import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Educational Focus',
    description: (
      <>
        Comprehensive educational content designed specifically for robotics students
        learning to integrate AI with physical robotic systems.
      </>
    ),
  },
  {
    title: 'Hands-on Learning',
    description: (
      <>
        Practical exercises and real-world examples that help you build actual
        humanoid robotics systems with AI capabilities.
      </>
    ),
  },
  {
    title: 'Modern Technologies',
    description: (
      <>
        Covers cutting-edge technologies including ROS 2, NVIDIA Isaac, Gazebo,
        and Vision-Language-Action systems.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}