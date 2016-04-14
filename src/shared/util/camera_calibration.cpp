#include "camera_calibration.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <limits>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "quaternion.h"
#include "field.h"
#include "field_default_constants.h"
#include "geomalgo.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

CameraParameters::CameraParameters(int camera_index_) :
    camera_index(camera_index_), p_alpha(Eigen::VectorXd(1)) {
  focal_length = new VarDouble("focal length", 500.0);
  principal_point_x = new VarDouble("principal point x", 390.0);
  principal_point_y = new VarDouble("principal point y", 290.0);
  distortion = new VarDouble("distortion", 0.0, 0.0, 2.0);
  //do not overwrite min/max ranges with values from config file
  distortion->addFlags(VARTYPE_FLAG_NOLOAD_ATTRIBUTES);

  q0 = new VarDouble("q0", 0.7);
  q1 = new VarDouble("q1", -0.7);
  q2 = new VarDouble("q2", .0);
  q3 = new VarDouble("q3", .0);

  tx = new VarDouble("tx", 0);
  ty = new VarDouble("ty", 1250);

  tz = new VarDouble("tz", 3500, 0, 5000);
  //do not overwrite min/max ranges with values from config file
  tz->addFlags(VARTYPE_FLAG_NOLOAD_ATTRIBUTES);

  q0P = new VarDouble("q0P", 0.7);
  q1P = new VarDouble("q1P", -0.7);
  q2P = new VarDouble("q2P", .0);
  q3P = new VarDouble("q3P", .0);

  txP = new VarDouble("txP", 0);
  tyP = new VarDouble("tyP", 1250);

  tzP = new VarDouble("tzP", 3500, 0, 5000);
  //do not overwrite min/max ranges with values from config file
  tzP->addFlags(VARTYPE_FLAG_NOLOAD_ATTRIBUTES);
  
  f2iToi2f();
  
  additional_calibration_information =
      new AdditionalCalibrationInformation(camera_index);

  q_rotate180 = Quaternion<double>(0, 0, 1.0,0);
}

CameraParameters::~CameraParameters() {
  delete focal_length;
  delete principal_point_x;
  delete principal_point_y;
  delete distortion;
  delete q0;
  delete q1;
  delete q2;
  delete q3;
  delete tx;
  delete ty;
  delete tz;
  delete q0P;
  delete q1P;
  delete q2P;
  delete q3P;
  delete txP;
  delete tyP;
  delete tzP;
  delete additional_calibration_information;
}

void CameraParameters::toProtoBuffer(
    SSL_GeometryCameraCalibration & buffer, int camera_id) const {
  buffer.set_focal_length(focal_length->getDouble());
  buffer.set_principal_point_x(principal_point_x->getDouble());
  buffer.set_principal_point_y(principal_point_y->getDouble());
  buffer.set_distortion(distortion->getDouble());
  buffer.set_q0(q0->getDouble());
  buffer.set_q1(q1->getDouble());
  buffer.set_q2(q2->getDouble());
  buffer.set_q3(q3->getDouble());
  buffer.set_tx(tx->getDouble());
  buffer.set_ty(ty->getDouble());
  buffer.set_tz(tz->getDouble());
  buffer.set_camera_id(camera_id);

  //--Set derived parameters:
  //compute camera world coordinates:
  Quaternion<double> q;
  q.set(q0->getDouble(),q1->getDouble(),q2->getDouble(),q3->getDouble());
  q.invert();

  GVector::vector3d<double> v_in(tx->getDouble(),ty->getDouble(),tz->getDouble());
  v_in=(-(v_in));

  GVector::vector3d<double> v_out = q.rotateVectorByQuaternion(v_in);
  buffer.set_derived_camera_world_tx(v_out.x);
  buffer.set_derived_camera_world_ty(v_out.y);
  buffer.set_derived_camera_world_tz(v_out.z);

}

GVector::vector3d< double > CameraParameters::getWorldLocation() {
  return GVector::vector3d<double>(txP->getDouble(),tyP->getDouble(),tzP->getDouble());
}

void CameraParameters::fromProtoBuffer(
    const SSL_GeometryCameraCalibration & buffer) {
  focal_length->setDouble(buffer.focal_length());
  principal_point_x->setDouble(buffer.principal_point_x());
  principal_point_y->setDouble(buffer.principal_point_y());
  distortion->setDouble(buffer.distortion());
  q0->setDouble(buffer.q0());
  q1->setDouble(buffer.q1());
  q2->setDouble(buffer.q2());
  q3->setDouble(buffer.q3());
  tx->setDouble(buffer.tx());
  ty->setDouble(buffer.ty());
  tz->setDouble(buffer.tz());
  
  f2iToi2f();
}

void CameraParameters::addSettingsToList(VarList& list) {
  list.addChild(focal_length);
  list.addChild(principal_point_x);
  list.addChild(principal_point_y);
  list.addChild(distortion);
  list.addChild(q0);
  list.addChild(q1);
  list.addChild(q2);
  list.addChild(q3);
  list.addChild(tx);
  list.addChild(ty);
  list.addChild(tz);
}

double CameraParameters::radialDistortion(double ru) const {
  if ((distortion->getDouble())<=DBL_MIN)
    return ru;
  double rd = 0;
  double a = distortion->getDouble();
  double b = -9.0*a*a*ru + a*sqrt(a*(12.0 + 81.0*a*ru*ru));
  b = (b < 0.0) ? (-pow(b, 1.0 / 3.0)) : pow(b, 1.0 / 3.0);
  rd = pow(2.0 / 3.0, 1.0 / 3.0) / b -
      b / (pow(2.0 * 3.0 * 3.0, 1.0 / 3.0) * a);
  return rd;
}

double CameraParameters::radialDistortion(double ru, double dist) const {
  if (dist<=DBL_MIN)
    return ru;
  double rd = 0;
  double a = dist;
  double b = -9.0*a*a*ru + a*sqrt(a*(12.0 + 81.0*a*ru*ru));
  b = (b < 0.0) ? (-pow(b, 1.0 / 3.0)) : pow(b, 1.0 / 3.0);
  rd = pow(2.0 / 3.0, 1.0 / 3.0) / b -
      b / (pow(2.0 * 3.0 * 3.0, 1.0 / 3.0) * a);
  return rd;
}

double CameraParameters::radialDistortionInv(double rd) const {
  double ru = rd*(1.0+rd*rd*distortion->getDouble());
  return ru;
}

void CameraParameters::radialDistortionInv(
    GVector::vector2d<double> &pu, const GVector::vector2d<double> &pd) const {
  double ru = radialDistortionInv(pd.length());
  pu = pd;
  pu = pu.norm(ru);
}

void CameraParameters::radialDistortion(
    const GVector::vector2d<double> pu, GVector::vector2d<double> &pd) const {
  double rd = radialDistortion(pu.length());
  pd = pu;
  pd = pd.norm(rd);
}

void CameraParameters::radialDistortion(
    const GVector::vector2d<double> pu,
    GVector::vector2d<double> &pd, double dist) const {
  double rd = radialDistortion(pu.length(), dist);
  pd = pu;
  pd = pd.norm(rd);
}

void CameraParameters::field2image(
    const GVector::vector3d<double> &p_f,
    GVector::vector2d<double> &p_i) const {

  //Convert R' to R
  Quaternion<double> q_field2cam = Quaternion<double>(
      q0P->getDouble(),q1P->getDouble(),q2P->getDouble(),q3P->getDouble());
  q_field2cam.invert();
  q_field2cam.norm();
  
  //Convert T' to T
  GVector::vector3d<double> translation = GVector::vector3d<double>(
      txP->getDouble(),tyP->getDouble(),tzP->getDouble());
  translation = q_field2cam.rotateVectorByQuaternion(GVector::vector3d<double>(0,0,0) - translation);
  
  // First transform the point from the field into the coordinate system of the
  // camera
  GVector::vector3d<double> p_c =
      q_field2cam.rotateVectorByQuaternion(p_f) + translation;
  GVector::vector2d<double> p_un =
      GVector::vector2d<double>(p_c.x/p_c.z, p_c.y/p_c.z);
  // Apply distortion
  GVector::vector2d<double> p_d;
  radialDistortion(p_un,p_d);

  if(p_c.z < 0) {
    p_d.x = -p_d.x;
    p_d.y = -p_d.y;
  }

  // Then project from the camera coordinate system onto the image plane using
  // the instrinsic parameters
  p_i = focal_length->getDouble() * p_d +
      GVector::vector2d<double>(principal_point_x->getDouble(),
                                principal_point_y->getDouble());
}

void CameraParameters::field2image(
    GVector::vector3d<double> &p_f, GVector::vector2d<double> &p_i,
    Eigen::VectorXd &p) {
  
  //Add increment to R'
  Quaternion<double> q_field2cam = Quaternion<double>(
      q0P->getDouble(),q1P->getDouble(),q2P->getDouble(),q3P->getDouble());
  q_field2cam.norm();
  GVector::vector3d<double> aa_diff(p[Q_1], p[Q_2], p[Q_3]);
  Quaternion<double> q_diff;
  q_diff.setAxis(aa_diff.norm(), aa_diff.length());
  q_field2cam = q_diff * q_field2cam;

  //Convert R' to R
  q_field2cam.invert();
  q_field2cam.norm();
  
  //Add increment to T'
  GVector::vector3d<double> translation = GVector::vector3d<double>(
      txP->getDouble(),tyP->getDouble(),tzP->getDouble());
  GVector::vector3d<double> t_diff(p[T_1], p[T_2], p[T_3]);
  translation = translation + t_diff;
  
  //Convert T' to T
  translation = q_field2cam.rotateVectorByQuaternion(GVector::vector3d<double>(0,0,0) - translation);
  
  // First transform the point from the field into the coordinate system of the
  // camera
  GVector::vector3d<double> p_c =
      q_field2cam.rotateVectorByQuaternion(p_f) + translation;
  GVector::vector2d<double> p_un =
      GVector::vector2d<double>(p_c.x/p_c.z, p_c.y/p_c.z);

  // Apply distortion
  GVector::vector2d<double> p_d;
  radialDistortion(p_un,p_d,distortion->getDouble() + p[DIST]);

  // Then project from the camera coordinate system onto the image plane using
  // the instrinsic parameters
  p_i = (focal_length->getDouble() + p[FOCAL_LENGTH]) * p_d +
      GVector::vector2d<double>(principal_point_x->getDouble() + p[PP_X],
                                principal_point_y->getDouble() + p[PP_Y]);
}

void CameraParameters::image2field(
    GVector::vector3d<double> &p_f, const GVector::vector2d<double> &p_i,
    double z) const {
  // Undo scaling and offset
  GVector::vector2d<double> p_d(
      (p_i.x - principal_point_x->getDouble()) / focal_length->getDouble(),
      (p_i.y - principal_point_y->getDouble()) / focal_length->getDouble());

  // Compensate for distortion (undistort)
  GVector::vector2d<double> p_un;
  radialDistortionInv(p_un,p_d);

  // Now we got a ray on the z axis
  GVector::vector3d<double> v(p_un.x, p_un.y, 1);

  // Transform this ray into world coordinates
  Quaternion<double> q_field2cam = Quaternion<double>(
      q0->getDouble(),q1->getDouble(),q2->getDouble(),q3->getDouble());
  q_field2cam.norm();
  GVector::vector3d<double> translation =
    GVector::vector3d<double>(tx->getDouble(),ty->getDouble(),tz->getDouble());

  Quaternion<double> q_field2cam_inv = q_field2cam;
  q_field2cam_inv.invert();
  GVector::vector3d<double> v_in_w =
      q_field2cam_inv.rotateVectorByQuaternion(v);
  GVector::vector3d<double> zero_in_w =
      q_field2cam_inv.rotateVectorByQuaternion(
          GVector::vector3d<double>(0,0,0) - translation);

  // Compute the the point where the rays intersects the field
  double t = GVector::ray_plane_intersect(
      GVector::vector3d<double>(0,0,z), GVector::vector3d<double>(0,0,1).norm(),
      zero_in_w, v_in_w.norm());

  // Set p_f
  p_f = zero_in_w + v_in_w.norm() * t;
}

void CameraParameters::i2fTof2i() {
  
  cout << "T': " << txP->getDouble() << " " << tyP->getDouble() << " " << tzP->getDouble() << endl;
  cout << "R': " << q0P->getDouble() << " " << q1P->getDouble() << " " << q2P->getDouble() << " " << q3P->getDouble() << endl;
  
  cout << "T: " << tx->getDouble() << " " << ty->getDouble() << " " << tz->getDouble() << endl;
  cout << "R: " << q0->getDouble() << " " << q1->getDouble() << " " << q2->getDouble() << " " << q3->getDouble() << endl;
  
  //Convert R^prime to R and T^prime to T
  Quaternion<double> q_f2i = Quaternion<double>(
	q0P->getDouble(),q1P->getDouble(),q2P->getDouble(),q3P->getDouble());
  q_f2i.invert();
  q_f2i.norm();
  GVector::vector3d<double> translation = GVector::vector3d<double>(txP->getDouble(), tyP->getDouble(), tzP->getDouble());
  translation = q_f2i.rotateVectorByQuaternion(GVector::vector3d<double>(0,0,0) - translation);
  q0->setDouble(q_f2i.x);
  q1->setDouble(q_f2i.y);
  q2->setDouble(q_f2i.z);
  q3->setDouble(q_f2i.w);
  tx->setDouble(translation.x);
  ty->setDouble(translation.y);
  tz->setDouble(translation.z);
  
  cout << "T: " << tx->getDouble() << " " << ty->getDouble() << " " << tz->getDouble() << endl;
  cout << "R: " << q0->getDouble() << " " << q1->getDouble() << " " << q2->getDouble() << " " << q3->getDouble() << endl;
}

void CameraParameters::f2iToi2f() {

  cout << "T: " << tx->getDouble() << " " << ty->getDouble() << " " << tz->getDouble() << endl;
  cout << "R: " << q0->getDouble() << " " << q1->getDouble() << " " << q2->getDouble() << " " << q3->getDouble() << endl;
  
  cout << "T': " << txP->getDouble() << " " << tyP->getDouble() << " " << tzP->getDouble() << endl;
  cout << "R': " << q0P->getDouble() << " " << q1P->getDouble() << " " << q2P->getDouble() << " " << q3P->getDouble() << endl;

  //Convert R to R^prime and T to T^prime
  Quaternion<double> q_c2f = Quaternion<double>(
	q0->getDouble(),q1->getDouble(),q2->getDouble(),q3->getDouble());
  q_c2f.invert();
  q_c2f.norm();
  GVector::vector3d<double> translation = GVector::vector3d<double>(tx->getDouble(), ty->getDouble(), tz->getDouble());
  translation = q_c2f.rotateVectorByQuaternion(GVector::vector3d<double>(0,0,0) - translation);
  q0P->setDouble(q_c2f.x);
  q1P->setDouble(q_c2f.y);
  q2P->setDouble(q_c2f.z);
  q3P->setDouble(q_c2f.w);
  txP->setDouble(translation.x);
  tyP->setDouble(translation.y);
  tzP->setDouble(translation.z);

  cout << "T': " << txP->getDouble() << " " << tyP->getDouble() << " " << tzP->getDouble() << endl;
  cout << "R': " << q0P->getDouble() << " " << q1P->getDouble() << " " << q2P->getDouble() << " " << q3P->getDouble() << endl;
}

double CameraParameters::calc_chisqr(
    std::vector<GVector::vector3d<double> > &p_f,
    std::vector<GVector::vector2d<double> > &p_i, Eigen::VectorXd &p) {
  assert(p_f.size() == p_i.size());

  double cov_cx_inv =
      1.0 / additional_calibration_information->cov_corner_x->getDouble();
  double cov_cy_inv =
      1.0 / additional_calibration_information->cov_corner_y->getDouble();

  double cov_lsx_inv =
      1.0 / additional_calibration_information->cov_ls_x->getDouble();
  double cov_lsy_inv =
      1.0 / additional_calibration_information->cov_ls_y->getDouble();

  double chisqr(0);

  // Iterate over manual points
  std::vector<GVector::vector3d<double> >::iterator it_p_f  = p_f.begin();
  std::vector<GVector::vector2d<double> >::iterator it_p_i  = p_i.begin();

  for (; it_p_f != p_f.end(); it_p_f++, it_p_i++)
  {
    GVector::vector2d<double> proj_p;
    field2image(*it_p_f, proj_p, p);
    chisqr += (proj_p.x - it_p_i->x) * (proj_p.x - it_p_i->x) * cov_cx_inv +
        (proj_p.y - it_p_i->y) * (proj_p.y - it_p_i->y) * cov_cy_inv;
  }

  return chisqr;
}

void CameraParameters::do_calibration(int cal_type) {
  std::vector<GVector::vector3d<double> > p_f;
  std::vector<GVector::vector2d<double> > p_i;

  AdditionalCalibrationInformation* aci = additional_calibration_information;

  for (int i = 0; i < AdditionalCalibrationInformation::kNumControlPoints;
      ++i) {
    p_i.push_back(GVector::vector2d<double>(
      aci->control_point_image_xs[i]->getDouble(),
      aci->control_point_image_ys[i]->getDouble()));
    p_f.push_back(GVector::vector3d<double>(
      aci->control_point_field_xs[i]->getDouble(),
      aci->control_point_field_ys[i]->getDouble(), 0.0));
  }

  if(cal_type == FOUR_POINT_INITIAL) {
    initialCalibration(p_f, p_i);
  } else {
    fullCalibration(p_f, p_i);
  }
}

void CameraParameters::reset() {
  focal_length->resetToDefault();
  principal_point_x->resetToDefault();
  principal_point_y->resetToDefault();
  distortion->resetToDefault();
  tx->resetToDefault();
  ty->resetToDefault();
  tz->resetToDefault();
  q0->resetToDefault();
  q1->resetToDefault();
  q2->resetToDefault();
  q3->resetToDefault();
  
  f2iToi2f();
}

void CameraParameters::initialCalibration(std::vector<GVector::vector3d<double> > &p_f, std::vector<GVector::vector2d<double> > &p_i) {
  assert(p_f.size() == p_i.size());

  p_to_est.clear();
  p_to_est.push_back(FOCAL_LENGTH);
  p_to_est.push_back(Q_1);
  p_to_est.push_back(Q_2);
  p_to_est.push_back(Q_3);
  p_to_est.push_back(T_1);
  p_to_est.push_back(T_2);

  double lambda(0.01);

  Eigen::VectorXd p(STATE_SPACE_DIMENSION);
  p.setZero();

  // Calculate first chisqr for all points using the start parameters
  double old_chisqr = calc_chisqr(p_f, p_i, p);

#ifndef NDEBUG
  std::cerr << "Chi-square: "<< old_chisqr << std::endl;
#endif
  
  // Create and fill corner measurement covariance matrix
  Eigen::Matrix2d cov_corner_inv;
  cov_corner_inv <<
      1 / additional_calibration_information->cov_corner_x->getDouble(), 0 , 0 ,
      1 / additional_calibration_information->cov_corner_y->getDouble();

  // Create and fill line segment measurement covariance matrix
  Eigen::Matrix2d cov_ls_inv;
  cov_ls_inv << 1 / additional_calibration_information->cov_ls_x->getDouble(),
      0 , 0 , 1 / additional_calibration_information->cov_ls_y->getDouble();

  int stateDimension = STATE_SPACE_DIMENSION;
  // Matrices for A, b and the Jacobian J
  Eigen::MatrixXd alpha(stateDimension,
                        stateDimension);
  Eigen::VectorXd beta(stateDimension, 1);
  Eigen::MatrixXd J(2, stateDimension);

  bool stop_optimization(false);
  int convergence_counter(0);
  double t_start=GetTimeSec();
  while (!stop_optimization) {
    // Calculate Jacobi-Matrix, alpha and beta
    // Iterate over alle point pairs
    std::vector<GVector::vector3d<double> >::iterator it_p_f  = p_f.begin();
    std::vector<GVector::vector2d<double> >::iterator it_p_i  = p_i.begin();
  
    double epsilon = sqrt(std::numeric_limits<double>::epsilon());

    alpha.setZero();
    beta.setZero();

    for (; it_p_f != p_f.end(); it_p_f++, it_p_i++) {
      J.setZero();

      GVector::vector2d<double> proj_p;
      field2image(*it_p_f, proj_p, p);
      proj_p = proj_p - *it_p_i;

      std::vector<int>::iterator it = p_to_est.begin();
      for (; it != p_to_est.end(); it++) {
        int i = *it;

        Eigen::VectorXd p_diff = p;
        p_diff(i) = p_diff(i) + epsilon;

        GVector::vector2d<double> proj_p_diff;
        field2image(*it_p_f, proj_p_diff, p_diff);
        J(0,i) = ((proj_p_diff.x - (*it_p_i).x) - proj_p.x) / epsilon;
        J(1,i) = ((proj_p_diff.y - (*it_p_i).y) - proj_p.y) / epsilon;
      }

      alpha += J.transpose() * cov_corner_inv * J;
      beta += J.transpose() * cov_corner_inv *
          Eigen::Vector2d(proj_p.x, proj_p.y);
    }
    
    // Augment alpha
    alpha += Eigen::MatrixXd::Identity(
        STATE_SPACE_DIMENSION, STATE_SPACE_DIMENSION)
        * lambda;

    // Solve for x
    Eigen::VectorXd new_p(STATE_SPACE_DIMENSION);

    // Due to an API change we need to check for
    // the right call at compile time
#ifdef EIGEN_WORLD_VERSION
    // alpha.llt().solve(-beta, &new_p); -- modify 1/15/16
    //  -- move to Eigen3 structure - 
    //  -- http://eigen.tuxfamily.org/dox/Eigen2ToEigen3.html
    new_p = alpha.llt().solve(-beta);
#else
    Eigen::Cholesky<Eigen::MatrixXd> c(alpha);
    new_p = c.solve(-beta);
#endif

    // Calculate chisqr again
    double chisqr = calc_chisqr(p_f, p_i, new_p);

    if (chisqr < old_chisqr) {
      focal_length->setDouble(focal_length->getDouble() + new_p[FOCAL_LENGTH]);
      principal_point_x->setDouble(
          principal_point_x->getDouble() + new_p[PP_X]);
      principal_point_y->setDouble(
          principal_point_y->getDouble() + new_p[PP_Y]);
      distortion->setDouble(distortion->getDouble() + new_p[DIST]);
      txP->setDouble(txP->getDouble() + new_p[T_1]);
      tyP->setDouble(tyP->getDouble() + new_p[T_2]);
      tzP->setDouble(tzP->getDouble() + new_p[T_3]);

      Quaternion<double> q_diff;
      GVector::vector3d<double> aa_diff(new_p[Q_1], new_p[Q_2], new_p[Q_3]);
      q_diff.setAxis(aa_diff.norm(), aa_diff.length());
      Quaternion<double> q_cam2field = Quaternion<double>(
          q0P->getDouble(),q1P->getDouble(),q2P->getDouble(),q3P->getDouble());
      q_cam2field = q_diff * q_cam2field;
      q_cam2field.norm();
      q0P->setDouble(q_cam2field.x);
      q1P->setDouble(q_cam2field.y);
      q2P->setDouble(q_cam2field.z);
      q3P->setDouble(q_cam2field.w);

      // Normalize focal length an orientation when the optimization tends to go into the wrong
      // of both possible projections
      if (focal_length->getDouble() < 0) {
        focal_length->setDouble(-focal_length->getDouble());
        q_cam2field = q_rotate180 * q_cam2field;
        q0P->setDouble(q_cam2field.x);
        q1P->setDouble(q_cam2field.y);
        q2P->setDouble(q_cam2field.z);
        q3P->setDouble(q_cam2field.w);
      }

      if (old_chisqr - chisqr < 0.001) {
        stop_optimization = true;
      } else {
        lambda /= 10;
        convergence_counter = 0;
      }

      old_chisqr = chisqr;
#ifndef NDEBUG
      std::cerr << "Chi-square: "<< old_chisqr << std::endl;
#endif
    } else {
      lambda *= 10;
      if (convergence_counter++ > 10) stop_optimization = true;
    }
    if ((GetTimeSec() - t_start) >
        additional_calibration_information->convergence_timeout->getDouble()) {
      stop_optimization=true;
    }
  }

  i2fTof2i();
  
// Debug output starts here
#ifndef NDEBUG

  // Estimated parameters
  std::cerr << "Estimated parameters: " << std::endl;
  std::cerr << focal_length->getDouble() << " "
            << principal_point_x->getDouble() << " "
            << principal_point_y->getDouble() << " " << distortion->getDouble()
            << std::endl;
  std::cerr << q0->getDouble() << " " << q1->getDouble() << " "
            << q2->getDouble() << " " << q3->getDouble() << std::endl;
  std::cerr << tx->getDouble() << " " << ty->getDouble() << " "
            << tz->getDouble() <<  std::endl;
  std::cerr << "alphas: " << p_alpha << std::endl;

  // Testing calibration by projecting the four field points into the image
  // plane and calculate MSE
  std::vector<GVector::vector3d<double> >::iterator it_p_f  = p_f.begin();
  std::vector<GVector::vector2d<double> >::iterator it_p_i  = p_i.begin();

  double corner_x(0);
  double corner_y(0);
  for (; it_p_f != p_f.end(); it_p_f++, it_p_i++) {
    GVector::vector2d<double> proj_p;
    field2image(*it_p_f, proj_p);
    GVector::vector3d<double> some_point;
    image2field(some_point,proj_p, 150);
    std::cerr << "Point in world: ("<< it_p_f->x << "," << it_p_f->y << ","
              << it_p_f->z  << ")" << std::endl;
    std::cerr << "Point should be at (" << it_p_i->x << "," << it_p_i->y
              << ") and is projected at (" << proj_p.x << "," << proj_p.y <<")"
              << std::endl;

    corner_x += (proj_p.x - it_p_i->x) * (proj_p.x - it_p_i->x);
    corner_y += (proj_p.y - it_p_i->y) * (proj_p.y - it_p_i->y);
  }

  std::cerr << "RESIDUAL CORNER POINTS: " << sqrt(corner_x/4) << " "
            << sqrt(corner_y/4) << std::endl;
	    
  #endif

}

struct ImageToFieldCostFunctor {
  ImageToFieldCostFunctor(GVector::vector2d<double> normal, double offset, GVector::vector2d<double> imagePoint, double height)
      : _imagePoint(imagePoint), _height(height) {
	_isStraightLine = true;
	_normal.x = normal.x;
	_normal.y = normal.y;
	_offset = offset;
      }

  ImageToFieldCostFunctor(GVector::vector3d<double> center, double radius, GVector::vector2d<double> imagePoint, double height)
      : _imagePoint(imagePoint), _height(height) {
	_isStraightLine = false;
	_centerOfArc.x = center.x;
	_centerOfArc.y = center.y;
	_centerOfArc.z = center.z;
	_radiusOfArc = radius;
      }
      
  template <typename T>
  bool operator()(const T* const camera_rotation,
                  const T* const camera_translation,
		  const T* const camera_intrinsics,
                  T* residuals) const {
  
    T oX = T(_imagePoint.x);
    T oY = T(_imagePoint.y);
    
    const T& pX = camera_intrinsics[0];
    const T& pY = camera_intrinsics[1];
    const T& focal = camera_intrinsics[2];
    const T& distortion = camera_intrinsics[3];

    if(focal <= T(0)) 
      return false;
    if(distortion < T(0)) 
      return false;
    if(pX < T(0))
      return false;
    if(pY < T(0))
      return false;
    
    T oPointInCamera[3];
    oPointInCamera[0] = (oX - pX) / focal;
    oPointInCamera[1] = (oY - pY) / focal;
    oPointInCamera[2] = T(1);
    
    T rd = ceres::sqrt(oPointInCamera[0] * oPointInCamera[0] + oPointInCamera[1] * oPointInCamera[1]);
    T ru = rd*(1.0+rd*rd*distortion);
    oPointInCamera[0] = oPointInCamera[0] * ru / rd;
    oPointInCamera[1] = oPointInCamera[1] * ru / rd;
    
    //cout << "oPointInCamera: x " << oPointInCamera[0] << " y " << oPointInCamera[1] << " z " << oPointInCamera[2] << endl;
    T oPointInWorld[3];
    ceres::QuaternionRotatePoint(camera_rotation, oPointInCamera, oPointInWorld);
    //cout << "oPointInWorld: x " << oPointInWorld[0] << " y " << oPointInWorld[1] << " z " << oPointInWorld[2] << endl;
    
    T zero_in_world[3] = {camera_translation[0], camera_translation[1], T(_height)};
    
    T len = ceres::sqrt(oPointInWorld[0] * oPointInWorld[0] + oPointInWorld[1] * oPointInWorld[1] + oPointInWorld[2] * oPointInWorld[2]);
    oPointInWorld[0] /= len;
    oPointInWorld[1] /= len;
    oPointInWorld[2] /= len;
    //cout << "oPointInWorld: x " << oPointInWorld[0] << " y " << oPointInWorld[1] << " z " << oPointInWorld[2] << endl;
    
    // (0,0,0), (0,0,1).norm(),zero_in_w, v_in_w.norm());
    // pOrigin, pNormal, rOrigin, rVector
    //return(dot(-pNormal,(rOrigin - pOrigin)) / (dot(pNormal,rVector)));
    //return(-zero_in_w[2] / v_in_w_norm[2]);
    T factor = -zero_in_world[2] / oPointInWorld[2];
    //cout << "factor: " << factor << endl;
    
    oPointInWorld[0] = zero_in_world[0] + oPointInWorld[0] * factor;
    oPointInWorld[1] = zero_in_world[1] + oPointInWorld[1] * factor;
    oPointInWorld[2] = zero_in_world[2] + oPointInWorld[2] * factor;
    //cout << "oPointInWorld:" << oPointInWorld[0] << "," << oPointInWorld[1] << "," << oPointInWorld[2] << endl;
    //cout << "aPointInWorld:" << field_x << "," << field_y << "," << field_z << endl;
    
    T error;
    if(_isStraightLine) {
      T offset = T(_normal.x) * oPointInWorld[0] + T(_normal.y) * oPointInWorld[1];
      // The error is the difference between the calculated offse	ts
      error = offset - _offset;
    } else {
      T radius = ceres::pow(T(_centerOfArc.x) - oPointInWorld[0], 2) + ceres::pow(T(_centerOfArc.y)  -oPointInWorld[1], 2) + ceres::pow(T(_centerOfArc.z) - oPointInWorld[2], 2);
      radius = ceres::sqrt(radius);
      // The error is the difference between the calculated offsets
      error = radius - T(_radiusOfArc);
    }
    residuals[0] = error;
    // z co ordinate of all field points should be 0
    //residuals[1] = oPointInWorld[2];
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(GVector::vector2d<double> normal, 
				     double offset,
				     const GVector::vector2d<double> imagePoint,
				     const double height
				    ) {
    return (new ceres::AutoDiffCostFunction<
            ImageToFieldCostFunctor, 1, 4, 2, 4>(
                new ImageToFieldCostFunctor(normal, offset, imagePoint, height)));
  }
  
  static ceres::CostFunction* Create(GVector::vector3d<double> center, 
				     double radius,
				     const GVector::vector2d<double> imagePoint,
				     const double height
				    ) {
    return (new ceres::AutoDiffCostFunction<
            ImageToFieldCostFunctor, 1, 4, 2, 4>(
                new ImageToFieldCostFunctor(center, radius, imagePoint, height)));
  }
  
  bool _isStraightLine;
  GVector::vector3d<double> _centerOfArc;
  GVector::vector2d<double> _normal;
  double _offset, _height, _radiusOfArc;
  GVector::vector2d<double> _imagePoint;
};

void CameraParameters::fullCalibration(std::vector<GVector::vector3d<double> > &p_f, std::vector<GVector::vector2d<double> > &p_i) 
{
  double camIntrinsics[4] = {principal_point_x->getDouble(), principal_point_y->getDouble(), focal_length->getDouble(), distortion->getDouble()};
  double camTranslation[3] = {txP->getDouble(), tyP->getDouble(), tzP->getDouble()};
  //ceres uses w,x,y,z order to represent quaternions
  double rotation[4] = {q3P->getDouble(), q0P->getDouble(), q1P->getDouble(), q2P->getDouble()};
  
  Problem problem;
  int numberOfBadPoints = 0;
  int numberOfGoodPoints = 0;
  int numberOfPoints = 0;
  
  std::vector<CalibrationData>::iterator ls_it = calibrationSegments.begin();
  for (; ls_it != calibrationSegments.end(); ls_it++) {
    
    GVector::vector2d<double> normal;
    double offset;
    //y = mx + c => y - mx = c => (-m)x + y = c => a = -m, b = 1, offset = c
    if((*ls_it).straightLine == true) {
      if((*ls_it).p1.y == (*ls_it).p2.y) {
	normal.x = 0;
	normal.y = 1;
	offset = (*ls_it).p1.y;
      } else if((*ls_it).p1.x == (*ls_it).p2.x) {
	normal.x = 1;
	normal.y = 0;
	offset = (*ls_it).p1.x;
      }
    }
    
    std::vector< std::pair<GVector::vector2d<double>,bool> >::iterator
	pts_it = (*ls_it).imgPts.begin();
    
    for (; pts_it != (*ls_it).imgPts.end(); pts_it++) {
      ++numberOfPoints;
      if (pts_it->second) {
	++numberOfGoodPoints;
	if((*ls_it).straightLine == true) {
	  CostFunction* cost_function = ImageToFieldCostFunctor::Create(normal, offset, pts_it->first, camTranslation[2]);
	  problem.AddResidualBlock(cost_function, NULL, rotation, camTranslation, camIntrinsics);
	} else {
	  CostFunction* cost_function = ImageToFieldCostFunctor::Create((*ls_it).center, (*ls_it).radius, pts_it->first, camTranslation[2]);
	  problem.AddResidualBlock(cost_function, NULL, rotation, camTranslation, camIntrinsics); 
	}
      } else {
	++numberOfBadPoints;
      }
    }
  }
  
  ceres::QuaternionParameterization *quaternion_parameterization = new ceres::QuaternionParameterization();
  problem.SetParameterization(rotation, new ceres::QuaternionParameterization());
  problem.SetParameterLowerBound(camIntrinsics, 3, 0);
  // Run the solver!
  Solver::Options options;
  
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  std::cout << "focalLength: " << camIntrinsics[2] << " px: " << camIntrinsics[0] << " py: " << camIntrinsics[1] << " distortion: " << camIntrinsics[3] << endl;
  std::cout << "q0P: " << rotation[1] << " q1P: " << rotation[2] << " q2P: " << rotation[3] << " q3P: " << rotation[0] << endl;
  std::cout << "cx: " << camTranslation[0] << " cy: " << camTranslation[1] << " cz: " << camTranslation[2] << endl;

  q0P->setDouble(rotation[1]);//x
  q1P->setDouble(rotation[2]);//y
  q2P->setDouble(rotation[3]);//z
  q3P->setDouble(rotation[0]);//w
  txP->setDouble(camTranslation[0]);
  tyP->setDouble(camTranslation[1]);
  tzP->setDouble(camTranslation[2]);
  
  principal_point_x->setDouble(camIntrinsics[0]);
  principal_point_y->setDouble(camIntrinsics[1]);
  focal_length->setDouble(camIntrinsics[2]);
  distortion->setDouble(camIntrinsics[3]);
  
  i2fTof2i();
}

CameraParameters::AdditionalCalibrationInformation::
    AdditionalCalibrationInformation(int camera_index_) :
    camera_index(camera_index_){

  for (int i = 0; i < kNumControlPoints; ++i) {
    ostringstream convert;
    convert << i;
    const string i_str = convert.str();
    control_point_set[i] = new VarList("Control Point " + i_str);
    control_point_names[i] = new VarString(
        "Control point " + i_str + " name", "Control point " + i_str);
    control_point_set[i]->addChild(control_point_names[i]);
    control_point_image_xs[i] = new VarDouble(
        "Control point " + i_str + " image x", 10.0);
    control_point_set[i]->addChild(control_point_image_xs[i]);
    control_point_image_ys[i] = new VarDouble(
        "Control point " + i_str + " image y", 10.0);
    control_point_set[i]->addChild(control_point_image_ys[i]);
    control_point_field_xs[i] = new VarDouble(
        "Control point " + i_str + " field x",
        FieldConstantsRoboCup2014::kCameraControlPoints[camera_index][i].x);
    control_point_set[i]->addChild(control_point_field_xs[i]);
    control_point_field_ys[i] = new VarDouble(
        "Control point " + i_str + " field y",
        FieldConstantsRoboCup2014::kCameraControlPoints[camera_index][i].y);
    control_point_set[i]->addChild(control_point_field_ys[i]);
  }

  initial_distortion = new VarDouble("initial distortion", 1.0);
  camera_height = new VarDouble("camera height", 4000.0);
  line_search_corridor_width = new VarDouble("line search corridor width",
                                             280.0);
  image_boundary = new VarDouble("Image boundary for edge detection", 10.0);
  max_feature_distance = new VarDouble("Max distance of edge from camera",
                                       9000.0);
  convergence_timeout = new VarDouble("convergence timeout (s)", 20.0);
  cov_corner_x = new VarDouble("Cov corner measurement x", 1.0);
  cov_corner_y = new VarDouble("Cov corner measurement y", 1.0);
  cov_ls_x = new VarDouble("Cov line segment measurement x", 1.0);
  cov_ls_y = new VarDouble("Cov line segment measurement y", 1.0);
  pointSeparation = new VarDouble("Points separation", 150);
}

CameraParameters::AdditionalCalibrationInformation::~AdditionalCalibrationInformation() {
  for (int i = 0; i < kNumControlPoints; ++i) {
    delete control_point_names[i];
    delete control_point_image_xs[i];
    delete control_point_image_ys[i];
    delete control_point_field_xs[i];
    delete control_point_field_ys[i];
  }

  delete initial_distortion;
  delete camera_height;
  delete line_search_corridor_width;
  delete image_boundary;
  delete max_feature_distance;
  delete convergence_timeout;
  delete cov_corner_x;
  delete cov_corner_y;
  delete cov_ls_x;
  delete cov_ls_y;
  delete pointSeparation;
}

void CameraParameters::AdditionalCalibrationInformation::addSettingsToList(
    VarList& list) {
  for (int i = 0; i < kNumControlPoints; ++i) {
    list.addChild(control_point_set[i]);
  }

  list.addChild(initial_distortion);
  list.addChild(camera_height);
  list.addChild(line_search_corridor_width);
  list.addChild(image_boundary);
  list.addChild(max_feature_distance);
  list.addChild(convergence_timeout);
  list.addChild(cov_corner_x);
  list.addChild(cov_corner_y);
  list.addChild(cov_ls_x);
  list.addChild(cov_ls_y);
  list.addChild(pointSeparation);
}
