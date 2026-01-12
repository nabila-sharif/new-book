import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className="text--center">
          <Link
            className="button button--secondary button--lg"
            to="/docs/physical-ai-humanoid-robotics">
            Start Learning - 30 min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Educational content for robotics students">
      <HomepageHeader />
      <main>
        <section className={styles.modulesSection}>
          <div className="container padding-horiz--md">
            <h2 className="text--center padding-top--md">Course Modules</h2>
            <div className="row">
              <div className="col col--4 padding--sm">
                <div className="card">
                  <div className="card__body text--center">
                    <h3>Module 1</h3>
                    <p>The Robotic Nervous System (ROS 2)</p>
                    <Link
                      className="button button--primary button--block"
                      to="/docs/physical-ai-humanoid-robotics">
                      Start Module 1
                    </Link>
                  </div>
                </div>
              </div>
              <div className="col col--4 padding--sm">
                <div className="card">
                  <div className="card__body text--center">
                    <h3>Module 2</h3>
                    <p>Digital Twin (Gazebo & Unity)</p>
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-2">
                      Start Module 2
                    </Link>
                  </div>
                </div>
              </div>
              <div className="col col--4 padding--sm">
                <div className="card">
                  <div className="card__body text--center">
                    <h3>Module 3</h3>
                    <p>The AI-Robot Brain (NVIDIA Isaac™)</p>
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-3">
                      Start Module 3
                    </Link>
                  </div>
                </div>
              </div>
            </div>
            <div className="row padding-top--md">
              <div className="col col--4 padding--sm col--offset-2">
                <div className="card">
                  <div className="card__body text--center">
                    <h3>Module 4</h3>
                    <p>Vision-Language-Action (VLA)</p>
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-4">
                      Start Module 4
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}